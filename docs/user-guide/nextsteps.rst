.. _data-curator-next-steps:

======================================
Next Steps
======================================

-----------------------------------------
Training a Tokenizer
-----------------------------------------
Tokenizers transform text into tokens that language models can interpret.
Training a tokenizer involves determining which strings of text should map to what token ids.
NeMo Curator does not handle training a tokenizer or tokenization in general, but NeMo does.
You can find more information on how to train a tokenizer using NeMo `here <https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/tokenizer/sentencepiece/train.html#training>`_.

-----------------------------------------
Training a Large Language Model
-----------------------------------------
Pretraining a large language model involves running next-token prediction on a large curated dataset.
NeMo handles everything for pretraining large language models using data curated with NeMo Curator.
You can find information on pretraining, evaluating, parameter efficient fine-tuning (PEFT) and more in the `large language model section of the NeMo user guide <https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/index.html#llm-index>`_.

-----------------------------------------
Aligning a Large Language Model
-----------------------------------------
After pretraining a large language model, aligning it allows you to interact with it in a chat-like setting.
NeMo Aligner allows you to take curated alignment data and use it to align a pretrained language model.
You can find information on how to use NeMo Aligner and all the alignment techniques it supports `here <https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/index.html>`_.