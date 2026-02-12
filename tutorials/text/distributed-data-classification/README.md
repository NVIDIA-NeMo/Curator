# Distributed Data Classification

The following is a set of Jupyter notebook tutorials which demonstrate how to use various text classification models supported by NeMo Curator.
The goal of using these classifiers is to help with data annotation, which is useful in data blending for foundation model training.

Each of these classifiers are available on Hugging Face and can be run independently with the [Transformers](https://github.com/huggingface/transformers) library.
By running them with NeMo Curator, the classifiers are accelerated using a heterogenous pipeline setup where tokenization is run across CPUs and model inference is run across GPUs.
Each of the Jupyter notebooks in this directory demonstrate how to run the classifiers on text data and are easily scalable to large amounts of data.

Before running any of these notebooks, see this [Installation Guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html#admin-installation) page for instructions on how to install NeMo Curator. Be sure to use an installation method which includes GPU dependencies.

For more information about the classifiers, refer to our [Distributed Data Classification](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/quality-assessment/distributed-classifier.html) documentation page.

## List of Classifiers

<div align="center">

| NeMo Curator Classifier | Description | Hugging Face Page |
| --- | --- | --- |
| `AegisClassifier` | Identify and categorize unsafe content per document | [nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0) and [nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0) |
| `ContentTypeClassifier` | Categorize the type-of-speech per document | [nvidia/content-type-classifier-deberta](https://huggingface.co/nvidia/content-type-classifier-deberta) |
| `DomainClassifier` | Categorize the domain per document | [nvidia/domain-classifier](https://huggingface.co/nvidia/domain-classifier) |
| `FineWebEduClassifier` | Determine the educational value per document; this model was trained using annotations from Llama 3 70B-Instruct | [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) |
| `FineWebMixtralEduClassifier` | Determine the educational value per document; this model was trained using annotations from Mixtral 8x22B-Instruct | [nvidia/nemocurator-fineweb-mixtral-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier) |
| `FineWebNemotronEduClassifier` | Determine the educational value per document; this model was trained using annotations from Nemotron-4-340B-Instruct | [nvidia/nemocurator-fineweb-nemotron-4-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier) |
| `InstructionDataGuardClassifier` | Identify LLM poisoning attacks per document | [nvidia/instruction-data-guard](https://huggingface.co/nvidia/instruction-data-guard) |
| `MultilingualDomainClassifier` | Categorize the domain per document; supports classification in 52 languages | [nvidia/multilingual-domain-classifier](https://huggingface.co/nvidia/multilingual-domain-classifier) |
| `PromptTaskComplexityClassifier` | Classifies text prompts across task types and complexity dimensions | [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) |
| `QualityClassifier` | Categorize documents as high, medium, or low quality | [quality-classifier-deberta](https://huggingface.co/nvidia/quality-classifier-deberta) |

</div>

Note that all classifiers support English text classification only, except the `MultilingualDomainClassifier`.

## Bring Your Own Classifier

Advanced users may want to integrate their own Hugging Face classifier(s) into NeMo Curator. Broadly, this requires creating a `CompositeStage` consisting of a CPU-based tokenizer stage and a GPU-based model inference stage. Refer to the [Text Classifiers README](https://github.com/NVIDIA-NeMo/Curator/tree/main/nemo_curator/stages/text/classifiers#text-classifiers) for details about how to do this.
