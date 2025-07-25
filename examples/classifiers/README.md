## Text Classification

The Python scripts in this directory demonstrate how to run classification on your text data with each of these classifiers:

- Domain Classifier
- Multilingual Domain Classifier
- Quality Classifier
- AEGIS Safety Models
- Instruction Data Guard Model
- FineWeb Educational Content Classifier
- FineWeb Mixtral Educational Classifier
- FineWeb Nemotron-4 Educational Classifier
- Content Type Classifier
- Prompt Task and Complexity Classifier

For more information about these classifiers, please see NeMo Curator's [Distributed Data Classification documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/distributeddataclassification.html).

Each of these scripts provide simple examples of what your own Python scripts might look like.

At a high level, you will:

1. Create a Dask client by using the `get_client` function
2. Use `DocumentDataset.read_json` (or `DocumentDataset.read_parquet`) to read your data
3. Initialize and call the classifier on your data
4. Write your results to the desired output type with `to_json` or `to_parquet`

Before running any of these scripts, we strongly recommend displaying `python <script name>.py --help` to ensure that any needed or relevant arguments are specified.
