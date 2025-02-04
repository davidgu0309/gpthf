from os import truncate
from re import L
import torch
from transformers.cache_utils import Cache


def thf_pooling_mask(batch_input_vector, wlt_embedding_width):
    B, d = batch_input_vector.shape  
    N = torch.max(batch_input_vector) + 1
    
    batch_tensor = torch.zeros(B, d, N+1, dtype=torch.bool, device=batch_input_vector.device)
    
    for i, input_vector in enumerate(batch_input_vector):
        valid_indices = input_vector != -1
        col_indices = input_vector[valid_indices]
        row_indices = torch.arange(d, device=batch_input_vector.device)[valid_indices]

        batch_tensor[i, row_indices, col_indices] = True
        batch_tensor[i] = torch.cummax(batch_tensor[i], dim=0).values

    batch_tensor = batch_tensor.repeat_interleave(wlt_embedding_width, dim=1)

    return batch_tensor

def generate_block_matrix(batch_vector):
    # Mark the positions of padded elements (-1)
    is_padded = batch_vector == -1

    # Determine change points for each sequence in the batch, ignoring transitions to padding
    change_mask = torch.cat([torch.ones_like(batch_vector[:, :1], dtype=torch.bool), batch_vector[:, :-1] != batch_vector[:, 1:]], dim=1)
    change_mask[is_padded] = False

    # Get the starting indices of each block for each sequence in the batch
    boundaries = change_mask.nonzero()
    #boundaries = vectorized_nonzero(change_mask)
    #assert torch.equal(boundaries, boundaries2)

    batch_boundaries_list = [boundaries[boundaries[:,0] == i][:,1].tolist() for i in range(batch_vector.shape[0])]

    # Find the maximum number of boundaries across all sequences, excluding padding
    max_boundaries = max(len(b) for b in batch_boundaries_list)

    # Pad each boundary list to the maximum length and construct the mask
    padded_boundaries = torch.full((len(batch_boundaries_list), max_boundaries), -1)
    boundary_mask = torch.zeros_like(padded_boundaries).bool()

    for i, boundary in enumerate(batch_boundaries_list):
        padded_boundaries[i, :len(boundary)] = torch.tensor(boundary)
        boundary_mask[i, :len(boundary)] = True

    # Expand dimensions for broadcasting
    vec_exp = batch_vector.unsqueeze(2)
    vec_transposed = batch_vector.unsqueeze(1)

    same_value_mask = (vec_exp == vec_transposed).bool()
    block_matrix_mask = ~same_value_mask

    return block_matrix_mask, padded_boundaries, boundary_mask

def fetch_sentence_embeddings(values, indices, width):
    batch_size, M = indices.shape
    _, L, D = values.shape
    batch_indices = torch.arange(batch_size, device=indices.device)[:, None, None].expand(-1, M, width)
    index_offsets = torch.arange(width, device=indices.device)[None, None, :].expand(batch_size, M, width)
    all_indices = indices[:, :, None] + index_offsets

    padding_mask = all_indices >= L
    all_indices.clamp_(max=L-1)
    fetched_values = values[batch_indices, all_indices]
    fetched_values[padding_mask] = 0
    fetched_values = fetched_values.reshape(batch_size, width*M, -1)  

    # The data in fetched_values with width 2 is gonna be aranged like [0,1,31,32,55,56...] with shape (B, width*M, D)
    return fetched_values 

def last_sentence_indices(indices, key_padding_mask):
    last_valid_indices = key_padding_mask.cumsum(dim=1).argmax(dim=1)
    indices2 = indices.clone()
    mask = indices == -1
    indices2[:, :-1] = indices[:, 1:] - 1
    indices2[:, -1] = last_valid_indices
    indices2[mask] = -1
    mask2 = (indices2 == -2)
    vector_expanded = last_valid_indices[:, None].expand(-1, indices2.shape[1])
    indices2[mask2] = vector_expanded[mask2]
    adjusted_indices = indices2.clone()
    valid_indices2 = indices2.clone()
    valid_indices2[indices2 == -1] = key_padding_mask.size(1)
    gathered_key_padding_mask = key_padding_mask.gather(1, valid_indices2.clamp(max=key_padding_mask.size(1)-1))
    invalid_mask = (indices2 != -1) & (gathered_key_padding_mask == 0)
    valid_positions = key_padding_mask.cumsum(dim=1)
    last_valid_indices_expanded = valid_positions.gather(1, indices2.clamp(min=0))
    replace_mask = invalid_mask & (indices2 >= 0)
    adjusted_indices[replace_mask] = (last_valid_indices_expanded - 1).clamp(min=0)[replace_mask]
    return adjusted_indices

def fetch_last_sentence_embeddings(values, indices, width, key_padding_mask):
    indices2 = last_sentence_indices(indices, key_padding_mask)
    return fetch_sentence_embeddings(values, indices2, width)

def inverse_fetch(values, indices, mask, N):
	batch_size, M, D = values.shape
	output = torch.ones(batch_size, N, D, device=values.device, dtype=values.dtype)
	batch_indices = torch.arange(batch_size)[:, None].expand(-1, M).to(indices.device)
	output[batch_indices[mask], indices[mask]] = values[mask]

	return output

def filled_inverse_fetch(values, indices, mask, N):
    # Prepare the output tensor
    B, M, D = values.shape
    output = torch.zeros(B, N, D, device=values.device, dtype=values.dtype)

    # Calculate new indices by repeating each index according to the gaps between them
    # Start by calculating the gaps to the next index (or to N for the last index)
    for b in range(B):  # Process each batch independently
        for i in range(0, M):  # Step through index pairs
            start = indices[b, i]  # Start of the slice
            if i == M-1 or indices[b, i + 1] == -1:
                end = N  # End of the slice if not -1
            else:
                end = indices[b, i + 1]  # Use end of 'values' if -1 or last index

            # Repeat the slice and fill into the output
            output[b, start:end] = values[b, i].unsqueeze(0).expand(end-start, -1)

            if end == N:
                break
    return output

def create_column_mask(indices, N):
    B, _ = indices.shape
    column_mask = torch.zeros(B, N, N, dtype=torch.bool, device=indices.device)
    mask = indices != -1
    
    # Generate a batch index for each valid index
    batch_indices = torch.arange(B, device=indices.device).unsqueeze(-1).expand_as(indices)
    
    # Filter out invalid indices
    valid_batch_indices = batch_indices[mask]
    valid_indices = indices[mask]

    # Use advanced indexing to update the column_mask in a vectorized manner
    column_mask[valid_batch_indices, :, valid_indices] = True

    return column_mask

def combine_attention_masks(attention_mask, key_padding_mask):
    # Assuming type bool, 1 equals keeping and 0 masking, and attention_mask with shape (B, N, N) and key_padding_mask with shape (B, N)
    B, N = attention_mask.shape[0], attention_mask.shape[1]
    key_padding_mask_expanded = key_padding_mask.unsqueeze(1).expand(B, N, N)
    combined_mask = attention_mask & key_padding_mask_expanded
    return combined_mask

def truncate_tensor(values, indices, padding_id, normalize=False, past_key_value_encoder=None):
    result = []
    result_key_cache_encoder = [[] for _ in range(len(past_key_value_encoder.key_cache))]  if past_key_value_encoder is not None else None
    result_value_cache_encoder = [[] for _ in range(len(past_key_value_encoder.key_cache))]  if past_key_value_encoder is not None else None

    for b in range(values.shape[0]):
        new_starting_index = torch.argmax(indices[b])
        new_idx = values[b][new_starting_index:]
        mask = new_idx != padding_id
        new_idx = new_idx[mask]
        if normalize:
            new_idx = new_idx - new_idx.min()
        result.append(new_idx) 

        if past_key_value_encoder is not None:
            for i in range(len(past_key_value_encoder.key_cache)):
                # Truncate the tensor from the new starting index
                truncated_key = past_key_value_encoder.key_cache[i][b][:, new_starting_index:]
                truncated_value = past_key_value_encoder.value_cache[i][b][:, new_starting_index:]
                masked_key = truncated_key[:, mask]
                masked_value = truncated_value[:, mask]
                result_key_cache_encoder[i].append(masked_key)
                result_value_cache_encoder[i].append(masked_value)

    new_length = max(len(ni) for ni in result)
    if past_key_value_encoder is not None:
        past_key_value_encoder.key_cache = [torch.stack([torch.nn.functional.pad(ni, (0, 0, 0, new_length - ni.shape[1]), value=0) for ni in result_key_cache_encoder[i]]) for i in range(len(result_key_cache_encoder))]
        past_key_value_encoder.value_cache = [torch.stack([torch.nn.functional.pad(ni, (0, 0, 0, new_length - ni.shape[1]), value=0) for ni in result_value_cache_encoder[i]]) for i in range(len(result_value_cache_encoder))]

    return torch.stack([torch.nn.functional.pad(ni, (0, new_length - len(ni)), value=padding_id) for ni in result])

def create_overwrite_mask(ranges_start, ranges_end, max_size) -> torch.Tensor:
    B = ranges_start.size(0)
    result = torch.zeros((B, max_size), dtype=torch.bool)
    idx = torch.arange(max_size).expand(B, max_size).to(ranges_start.device)
    result = (idx >= ranges_start.unsqueeze(1)) & (idx < ranges_end.unsqueeze(1))
    return result

def overwrite_with_mask(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, N, D = a.size()
    if not a.is_contiguous():
        a = a.contiguous()
    a_flat = a.view(B*N, D)
    mask_flat = mask.view(-1)
    indices = torch.nonzero(mask_flat, as_tuple=True)[0]
    a_flat[indices] = b
    a_overwritten = a_flat.view_as(a)
    return a_overwritten

def prepare_inputs_for_generation(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs):
        #code stolen from transformers.models.llama.modeling_llama, who tied this function to a model instance  some
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        position_ids = position_ids[:, -input_ids.shape[1] :] if position_ids is not None else None

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
   