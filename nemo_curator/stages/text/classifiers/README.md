# Text Classifiers

The following files contain implementations for all of our text classifiers. In this document, we give an overview of common functionalities between all classifiers.

All of the underlying classes needed to implement a text classifier can be found in the `../models` directory and `base.py` script. We implement a classifier by making it a `CompositeStage` which expects a `DocumentBatch` as input and outputs a `DocumentBatch` with the added prediction column(s), along with a probability column if specified. In order to use a classifier, the user must specify a `text_field` (default value is `"text"`), which is a column in the input `DocumentBatch` that will be fed into the tokenizer and used to run inference. In `base.py`, we use the `DistributedDataClassifier` class as a commonly shared `CompositeStage` for various classifiers, since many of our classifiers are implemented in nearly identical ways. Classifiers which do not use the `DistributedDataClassifier` and instead implement their own composite stages include the `AegisClassifier`, `InstructionDataGuardClassifier`, `_FineWebBaseClassifier` (which is the parent class for our 3 FineWeb-based classifiers), and `PromptTaskComplexityClassifier`.

In the following sections, we break down each `ProcessingStage` used in the classification pipeline.

### Tokenizer Stage

Every classifier expects the text column to be tokenized before feeding it into the model inference stage. To do this, we use the `TokenizerStage`, which is found in `../models/tokenizer.py` and shared between all classifiers. It is initialized with information about the tokenizer including its Hugging Face identifier, Hugging Face token (if needed), the `text_field` to be tokenized, and other parameters about how the tokenizer should be configured (`max_seq_length`, `padding_size`, and `unk_token`).

The `setup_on_node` function is used to download the tokenizer and model locally, while the `setup` function is used to load the tokenizer per actor. Tokenization is done entirely on the CPU, so it uses the default `Resource` specification which underlies all `ProcessingStage` classes (`Resources(cpus=1)`).

Tokenization occurs in the `process` function. The `max_chars` parameter is used to slice the text before inputting it into the tokenizer, if desired. Then, we use `batch_encode_plus` to tokenize multiple texts at a time. Finally, we use the `sort_by_length` boolean to determine whether or not to sort the output tokens by their length. Sorting is recommended as it can lead to better performance in the model inference stage, so this parameter defaults to `True`. This creates a column called `"_curator_seq_order"` so that we can unsort to the original order of the input data at the end of the pipeline.

Please refer to `base.py` as an example of how the `TokenizerStage` can be used as part of the `CompositeStage` making up the base `DistributedDataClassifier` class.

### Model Stage

Every classifier builds upon the `ModelStage` class found in `../models/model.py`, which expects the `input_ids` and `attention_mask` values from the tokenization stage. Like the tokenizer stage, it is initialized with the relevant Hugging Face identifier and Hugging Face token (if needed). Often, the tokenizer stage and model stage use the same Hugging Face identifier, in which case the `setup_on_node` function does not need to redownload. We are assuming all models using the `HFModel` implementation can fit onto a single GPU, so we specify `Resources(cpus=1, gpus=1)`.

Similar to the tokenizer stage, the `setup` function is used to load the model per actor. Generically, the `setup` function is expected to load the model by using a custom class which extends the `nn.Module` and `PyTorchModelHubMixin` classes. This will consist of using `from_pretrained` to load the model. It will also involve overriding the `forward` function as needed (for example, we always override the `forward` function to enable the `autocast` boolean parameter for better model performance).

The core functions of the `ModelStage` class are `yield_next_batch` and `process`. The former uses the `model_inference_batch_size` parameter (which can be specified for the user but for which we have tried to set reasonable default values per classifier) to create digestible batch sizes for the model forward pass, as well as clips the tokens to remove unnecessary padding and help conserve as much memory as possible. When the forward pass into the model is done, we use `torch.no_grad` and several `del` calls to explicitly help manage memory.

The model outputs are processed with the `process_model_output`, `collect_outputs`, and `create_output_dataframe` functions:

- The `process_model_output` function is called immediately after the forward pass and is used to format the input tensors into a dictionary of NumPy arrays per batch. The result of this function is appended to a list called `processed_outputs`.
- The `collect_outputs` function is called after all forward pass results have been run and stored in `processed_outputs`. It combines all batches to create a dictionary with the column name(s) as the key(s) and the NumPy arrays as the values. This function should be model agnostic, assuming that the output of `process_model_output` is correctly formatted.
- The `create_output_dataframe` function is used to add the desired columns to the final output DataFrame.

If `sort_by_length` was `True` in the tokenizer stage, then the `has_seq_order` boolean should also be `True` so that the `"_curator_seq_order"` column can be used to unsort the DataFrame back to its original order.

Finally, the `DocumentBatch` with the predictions is formatted and returned. The `ModelStage` has an explicit `teardown` function to help clean up PyTorch cache and garbage collection.

Please refer to `base.py` as an example of how the `ModelStage` can be used as part of the `CompositeStage` making up the base `DistributedDataClassifier` class.

### Filter Stage (Optional)

Most classifiers support filtering after classification. This is particularly useful if the user is curating their data for domain-specific needs, wants to separate their data by a quality metric, or wants to filter out toxic content. For classifiers, the user may specify the `filter_by` parameter to tell the classifier to only keep records satisfying one or more labels. Whenever `filter_by` is not none, we add a `Filter` stage to the classification pipeline to achieve this.

Please refer to `base.py` as an example of how a `Filter` stage can be used as part of the `CompositeStage` making up the base `DistributedDataClassifier` class.

### Additional Stages (Optional)

For users developing their own classification pipelines, they may add additional stages to the `CompositeStage`, depending on their model. For example, our Aegis classifiers (in `aegis.py`) include text preprocessing and postprocessing stages, which occur before the tokenization stage and after the model inference stage, respectively. The goal of additional stages should be to reduce the user lift as much as possible.
