import torch
from torch import nn
import math
from .utils import fetch_sentence_embeddings

import torch.nn as nn

class THFClassifier(nn.Module):
    def __init__(self, config):
        super(THFClassifier, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim)
        #self.segment_embeddings = nn.Embedding(2, config.dim)
        wlt_encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim, nhead=config.wlt_encoder_num_attention_heads, dim_feedforward=config.wlt_encoder_hidden_dim, dropout=config.dropout, activation=config.activation, batch_first=True)
        self.wlt_encoder = nn.TransformerEncoder(wlt_encoder_layer, num_layers=config.wlt_encoder_num_hidden_layers)

        body_dimension = config.wlt_embedding_width * config.dim
        slt_body_layer = nn.TransformerEncoderLayer(d_model=body_dimension, nhead=config.slt_body_num_attention_heads, dim_feedforward=config.slt_body_hidden_dim, dropout=config.dropout, activation=config.activation, batch_first=True)
        self.slt_body = nn.TransformerEncoder(slt_body_layer, num_layers=config.slt_body_num_hidden_layers)

        wlt_decoder_layer = nn.TransformerEncoderLayer(d_model=config.dim, nhead=config.wlt_decoder_num_attention_heads, dim_feedforward=config.wlt_decoder_hidden_dim, dropout=config.dropout, activation=config.activation, batch_first=True)
        self.wlt_decoder = nn.TransformerEncoder(wlt_decoder_layer, num_layers=config.wlt_decoder_num_hidden_layers)

        self.pooler = nn.Linear(config.dim, config.dim)
        self.pooler_activation = nn.Tanh()
        self.classifier = nn.Linear(config.dim, 2)
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.position_encoding = torch.zeros((1, config.max_len, config.dim), device=self.device)
        X = torch.arange(config.max_len, dtype=torch.float32).reshape(-1, 1) \
            / torch.pow(10000, torch.arange(0, config.dim, 2, dtype=torch.float32) / config.dim)
        self.position_encoding[:, :, 0::2] = torch.sin(X)
        self.position_encoding[:, :, 1::2] = torch.cos(X)

        X = torch.arange(config.max_body_len, dtype=torch.float32).reshape(-1, 1) \
            / torch.pow(10000, torch.arange(0, config.dim, 2, dtype=torch.float32) / config.dim)
        
        self.sentence_position_encoding = torch.zeros((1, config.max_body_len, config.dim)).to(self.device)
        self.sentence_position_encoding[:, :, 0::2] = torch.sin(X)
        self.sentence_position_encoding[:, :, 1::2] = torch.cos(X)
        # self.position_encodings = nn.Embedding(config.max_len, config.dim)
        # self.sentence_position_encodings = nn.Embedding(config.max_body_len, config.dim)

    def forward(self, input_ids, attention_mask, mode, word_attention_mask, sentence_indices, sentence_attention_mask, segment_ids, every_n=None):
        batch_size, sent_len = input_ids.shape

        position_encoding = self.position_encoding[:, :sent_len, :].expand(batch_size, sent_len, -1) # (batch_size, sent_len, dim)
        #position_encoding = self.position_encodings(torch.arange(sent_len, device=self.device)).unsqueeze(0)
        x = position_encoding + self.word_embeddings(input_ids)

        if mode == 'regular':
            x = self.wlt_encoder(x, src_key_padding_mask=~attention_mask)
            sent_num = math.ceil(sent_len / every_n)
            sentence_embeddings = x[:, ::every_n, :]
            sentence_attention_mask = attention_mask[:, ::every_n]

        else:
            word_attention_mask = word_attention_mask.unsqueeze(1).expand(-1, self.config.wlt_encoder_num_attention_heads, -1, -1).reshape(-1, sent_len, sent_len) # expand to the number of heads
            x = self.wlt_encoder(x, src_key_padding_mask=~attention_mask, mask=word_attention_mask)
            sent_num = sentence_attention_mask.size(-1)
            sentence_embeddings = fetch_sentence_embeddings(x, sentence_indices)

        sentence_position_encoding = self.sentence_position_encoding[:, :sent_num*self.config.wlt_embedding_width, :].expand(batch_size, sent_num*self.config.wlt_embedding_width, -1) # (batch_size, sent_num*wlt_embedding_width, dim)
        #sentence_position_encoding = self.sentence_position_encodings(torch.arange(sent_num*self.config.wlt_embedding_width, device=self.device)).unsqueeze(0)
        sentence_embeddings = sentence_embeddings + sentence_position_encoding
            
        s = self.slt_body(sentence_embeddings, src_key_padding_mask=~sentence_attention_mask) # (batch_size, wlt_embedding_width * dim)
        y = torch.nn.functional.pad(s, (0, 0, 0, sent_len - s.size(-2)), mode='constant', value=1.0) + position_encoding

        #new_attention_mask = torch.nn.functional.pad(sentence_attention_mask, (0, sent_len - sentence_attention_mask.shape[1]), mode='constant', value=False)
        #y = self.wlt_decoder(y, src_key_padding_mask=~new_attention_mask) 

        y = self.wlt_decoder(y, src_key_padding_mask=~attention_mask) 
        
        first_token_tensor = y[:, 0]
        logits = self.pooler(first_token_tensor)
        pooled_output = self.pooler_activation(self.pooler(first_token_tensor))
        logits = self.classifier(pooled_output)
        return logits

