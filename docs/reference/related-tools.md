---
description: "Overview of NVIDIA AI ecosystem tools that integrate with NeMo Curator for end-to-end model development workflows"
categories: ["integrations-apis"]
tags: ["nemo-framework", "model-training", "tokenization", "alignment", "deployment", "enterprise"]
personas: ["mle-focused", "data-scientist-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(reference-tools)=
# NVIDIA AI Ecosystem: Related Tools

After preparing your data with NeMo Curator, you'll likely want to use it to train models. NVIDIA provides an integrated ecosystem of AI tools that work seamlessly with data prepared by NeMo Curator. This guide outlines the related tools for your next steps.

## NeMo Framework

[NVIDIA NeMo](https://github.com/NVIDIA/NeMo) is an end-to-end framework for building, training, and fine-tuning GPU-accelerated language models. It provides:

- Pretrained model checkpoints
- Training and inference scripts
- Optimization techniques for large-scale deployments

### Training a Tokenizer

Tokenizers transform text into tokens that language models can interpret. While NeMo Curator doesn't handle tokenizer training or tokenization in general, NeMo does.

Learn how to train a tokenizer using NeMo in the [tokenizer training documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/tokenizer/sentencepiece/train.html#training).

### Training Large Language Models

Pretraining a large language model involves running next-token prediction on large curated datasets, exactly the type that NeMo Curator helps you prepare. NeMo handles everything for pretraining large language models using your curated data.

Find comprehensive information on:
- Pretraining methodologies
- Model evaluation
- Parameter-efficient fine-tuning (PEFT)
- Distributed training

In the [large language model section of the NeMo user guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/index.html#llm-index).

## NeMo Aligner

[NVIDIA NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner) is a framework designed for aligning language models with human preferences.

After pretraining a large language model, aligning it allows you to interact with it in a chat-like setting. NeMo Aligner lets you take curated alignment data and use it to align a pretrained language model.

Learn about NeMo Aligner's capabilities including:
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO)
- Constitutional AI (CAI)

In the [NeMo Aligner documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/index.html).

## NVIDIA AI Enterprise

For organizations looking to deploy trained models to production, [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/) provides a software platform that includes enterprise support for:

- The complete NeMo framework
- Pretrained foundation models
- Deployment and inference tools
- Enterprise-grade security and support

## Complete Workflow

A typical end-to-end workflow with NVIDIA's AI tools includes:

1. **Data Preparation**: Use NeMo Curator to clean, filter, and prepare your dataset
2. **Tokenization**: Train or use a tokenizer with NeMo
3. **Model Training**: Pretrain or fine-tune models with NeMo
4. **Alignment**: Align models with human preferences using NeMo Aligner
5. **Deployment**: Deploy models using NVIDIA AI Enterprise or Triton Inference Server

This integrated ecosystem allows you to move from raw data to deployed, production-ready models with consistent tooling and optimized performance. 