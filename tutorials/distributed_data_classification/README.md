# Distributed Data Classification
The following is a set of Jupyter notebook tutorials which demonstrate how to use various text classification models supported by NeMo Curator.
The goal of using these classifiers is to help with data annotation, which is useful in data blending for foundation model training.

Each of these classifiers are available on Hugging Face and can be run independently with the [Transformers](https://github.com/huggingface/transformers) library.
By running them with NeMo Curator, the classifiers are accelerated using [CrossFit](https://github.com/rapidsai/crossfit), a library that leverages intellegent batching and RAPIDS to accelerate the offline inference on large datasets.
Each of the Jupyter notebooks in this directory demonstrate how to run the classifiers on text data and are easily scalable to large amounts of data.

Before running any of these notebooks, please see this [Getting Started](https://github.com/NVIDIA/NeMo-Curator?tab=readme-ov-file#get-started) page for instructions on how to install NeMo Curator.

## List of Classifiers

<div align="center">

| NeMo Curator Classifier | Hugging Face Page |
| --- | --- |
| `AegisClassifier` | [nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0) and [nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0) |
| `ContentTypeClassifier` | [nvidia/content-type-classifier-deberta](https://huggingface.co/nvidia/content-type-classifier-deberta) |
| `DomainClassifier` | [nvidia/domain-classifier](https://huggingface.co/nvidia/domain-classifier) |
| `FineWebEduClassifier` | [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) |
| `FineWebMixtralEduClassifier` | [nvidia/nemocurator-fineweb-mixtral-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier) |
| `FineWebNemotronEduClassifier` | [nvidia/nemocurator-fineweb-nemotron-4-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier) |
| `InstructionDataGuardClassifier` | [nvidia/instruction-data-guard](https://huggingface.co/nvidia/instruction-data-guard) |
| `MultilingualDomainClassifier` | [nvidia/multilingual-domain-classifier](https://huggingface.co/nvidia/multilingual-domain-classifier) |
| `PromptTaskComplexityClassifier` | [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) |
| `PyTorchClassifier` | Requires local .pth file(s) for any DeBERTa-based text classifier(s) |
| `QualityClassifier` | [quality-classifier-deberta](https://huggingface.co/nvidia/quality-classifier-deberta) |

</div>
