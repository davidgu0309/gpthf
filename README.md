# Text Compression for Efficient Language Generation

This repository contains the implementation and supplementary materials for the paper:
**"Text Compression for Efficient Language Generation"**. The paper associated with this repository is currently under review. A link to the full paper will be provided once the review process is complete.

In this work, we introduce the **Generative Pretrained Thought-former (GPTHF)**, a hierarchical transformer capable of text generation by compressing sentences into embeddings and employing a sentence-level attention mechanism. Our approach leads to significant efficiency improvements in FLOPs and runtime over traditional GPT-style models.

## Abstract

We challenge the prevailing assumption that LLMs must rely fully on sub-word tokens for high-quality text generation. To this end, we propose the ``Generative Pretrained Thoughtformer'' (GPTHF), a hierarchical transformer language model capable of text generation by compressing text into sentence embeddings and employing a sentence attention mechanism. GPTHF retains GPT’s architecture, modifying only token interactions via dynamic sparse attention masks. 

Our experiments show that GPTHF achieves an up to an order of magnitude improvement in FLOPs efficiency and a threefold increase in runtime speed compared to equally-sized GPT models in the low-size regime. This is achieved through a unique generation method that caches and reuses sentence embeddings, allowing significant portions of the input to bypass large parts of the network.

## Model overview

GPTHF consists of:

- A **word-level transformer encoder** (`wlt_encoder`) that compresses each sentence into a single embedding.
- A **sentence-level transformer body** (`slt_body`) that contextualizes sentence embeddings.
- A **fast generation algorithm** that caches and reuses sentence embeddings to improve efficiency.

![GPTHF Architecture](figures/architecture.png)

More details can be found in the paper (link provided once the review process is complete).

## Acknowledgments

This repository builds on the framework from (https://github.com/JonasGeiping/cramming), a framework for training language models with limited compute.
