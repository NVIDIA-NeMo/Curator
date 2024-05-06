
.. _data-curator-qualityfiltering:

============================================
Classifier and Heuristic Quality Filtering
============================================

-----------------------------------------
Background
-----------------------------------------

Large datasets often contain many documents considered to be "low quality".
In this context, "low quality" data simply means data we don't want a downstream model to learn from, and "high quality" data is data that we do want a downstream model to learn from.
The metrics that define quality can vary.
There are heuristics that measure quality by gathering simple statistics like how many punctutation marks a document has, how long is the document, and how repetitive is the document.
You can then filter documents by these statistics.
In contrast, you may have a high quality collection of data that you want a new dataset to align with.
You could train a simple classifier to differentiate between documents that look similar to those high quality documents and documents that do not.

NeMo Curator offers modules for both kinds of filtering, and it provides an easy interface for adding your own filters and combining them with existing ones.
You can also use these modules to collect statistics and metadata on your documents without removing any of them.
There are 30+ filters available for English, non-English, and code datasets.

-----------------------------------------
Usage
-----------------------------------------

The ``ScoreFilter`` is at the center of the filtering in NeMo Curator.
Let's examine this small example:

.. code-block:: python

    import nemo_curator as nc
    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.utils.file_utils import get_all_files_paths_under
    from nemo_curator.filters import WordCountFilter

    files = get_all_files_paths_under("books_dataset/")
    books = DocumentDataset.read_json(files, add_filename=True)

    filter_step = nc.ScoreFilter(
                    WordCountFilter(min_words=80),
                    text_field="text",
                    score_field="word_count",
                )

    long_books = filter_step(books)

    long_books.to_json("long_books/", write_to_filename=True)

The central part to focus on is the creation of the ``filter_step``.
``WordCountFilter(min_words=80)`` creates and configures a filter object.
A filter object is a class that inherits from the abstract base class ``nemo_curator.filters.DocumentFilter``.
This base class requires the inheritor to implement two methods, ``score_document`` and ``keep_document``.
For this example, let's look at a simplified version of the ``WordCountFilter``.

.. code-block:: python

  class WordCountFilter(DocumentFilter):

    def __init__(self, min_words=50, max_words=100000, lang='en'):
      self._min_words = min_words
      self._max_words = max_words
      self._word_splitter = get_word_splitter(lang)
      self._name = 'word_count'

    def score_document(self, text: str):
      return len(self._word_splitter(text))

    def keep_document(self, score: int):
      return self._min_words <= score <= self._max_words

With this implementation, it becomes clear what each function is doing.
``score_document`` takes the text of a document, and returns the number of words in the document.
``keep_document`` takes in the score outputted by ``score_document`` (the number of words in this case) and returns ``True`` if the score indicates the document should be kept and ``False`` if the document should be removed.
Now, it's important to note that ``WordCountFilter`` and ``DocumentFilter`` only operate on a single document.
In order to apply the filter to a whole dataset, we must use ``ScoreFilter``.

.. code-block:: python

  filter_step = nc.ScoreFilter(
      WordCountFilter(min_words=80),
      text_field="text",
      score_field="word_count",
  )

The construction of ``ScoreFilter`` creates a function that can be applied to a ``DocumentDataset`` instead of just a single document.
``text_field`` designates the field in the dataset that holds the documents that should get passed to the filter's ``score_document`` function.
``score_field`` is an optional argument that allows you to record the score in the given metadata field of the document, and if specified, it will be written to disk with the rest of the metadata.

In some cases, the dataset may come with metadata that you want to filter directly. Or, you might want to simply add a new piece of metadata without filtering on it.
The ``Filter`` and ``Score`` modules allow you to accomplish each task respectively.

For example, if the dataset in the above example came pre-populated with the ``word_count`` field, you could rewrite it as follows:

.. code-block:: python

    books = DocumentDataset.read_json(files, add_filename=True)

    filter_step = nc.Filter(
                    WordCountFilter(min_words=80).keep_document,
                    filter_field="word_count",
                )

    long_books = filter_step(books)

    long_books.to_json("long_books/", write_to_filename=True)

Alternatively, if you simply want to track the length of the words in the documents and not filter based on them, you could rewrite it as follows:

.. code-block:: python

    books = DocumentDataset.read_json(files, add_filename=True)

    filter_step = nc.Score(
                    WordCountFilter(min_words=80).score_document,
                    text_field="text",
                    score_field="word_count",
                )

    annotated_books = filter_step(books)

    annotated_books.to_json("annotated_books/", write_to_filename=True)


############################
Batched Filtering
############################

While the scoring and filtering functions defined above operate on single documents, NeMo Curator can take advantage of functions that operate in batches for improved performance.
To accomplish this, you can annotate your functions with the ``batched`` decorator.
This decorator will cause a pandas series of documents/scores to be passed to the function instead of a single document/score.
Here is the ``WordCountFilter`` rewritten to use batches in the ``keep_document``.

.. code-block:: python

  from nemo_curator.utils.decorators import batched

  class WordCountFilter(DocumentFilter):

    def __init__(self, min_words=50, max_words=100000, lang='en'):
      self._min_words = min_words
      self._max_words = max_words
      self._word_splitter = get_word_splitter(lang)
      self._name = 'word_count'

    def score_document(self, text: str):
      return len(self._word_splitter(text))

    @batched
    def keep_document(self, scores: pd.Series):
      pass_min = self._min_words <= scores
      pass_max = score <= self._max_words
      return pass_min & pass_max

When you use the ``batched`` decorator, the index of the series returned from the function must remain the same as the index that was passed in.
The index may not be continuous due to filters being applied prior to the current filter.
In the above code, the index will be the same automatically so no change is required.
However, when writing functions that transform the series into a different structure like a list, special care is needed.
The following code example demonstrates what this error may look like, and how to fix it.

.. code-block:: python

  class BuggyLengthFilter(DocumentFilter):

    @batched
    def score_document(self, documents: pd.Series):
      scores = []
      for document in documents:
        scores.append(len(document))

      return pd.Series(scores) # Bad! Does not preserve the index

  class CorrectLengthFilter(DocumentFilter):

    @batched
    def score_document(self, documents: pd.Series):
      scores = []
      for document in documents:
        scores.append(len(document))

      return pd.Series(scores, index=documents.index) # Good! Preserves the index


-----------------------------------------
Classifier Filtering
-----------------------------------------

The classifier-based filtering approach we have implemented follows closely to that used in `Brown et al., 2020 <https://arxiv.org/abs/2005.14165>`_,
and trains a binary skip-gram classifier that can be used to distinguish between low and high quality documents. To implement this, we use the
functions provided by fastText. Following the examples provided in the fastText documentation, we first create a file consisting of
high and low-quality training documents. We provide an example of how to train and use a model in ``examples/classifier_filtering.py``.

We also provide CLI scripts for the same functionality. The :code:`prepare_fasttext_training_data` script will randomly sample documents
from an input dataset and will prepare them to be used to train a fasText skip-gram classifier. For a high-quality dataset we recommend sampling from
either OpenWebText2 or Wikipedia and an unfiltered version of Common Crawl can be used for a low-quality dataset.

.. code-block:: bash

    prepare_fasttext_training_data \
      --input-data-dir=<Specify the path to common-crawl/low-quality data> \
      --output-num-samples=<Specify the number of low-quality documents to be used for training> \
      --label='__label__cc' \
      --output-train-file=${res_dir}/cc_samples.txt \

    prepare_fasttext_training_data \
      --input-data-dir=<Specify the path to high-quality data> \
      --output-num-samples=<Specify the number of high-quality documents to be used for training> \
      --label='__label__hq' \
      --output-train-file=${res_dir}/hq_samples.txt \

Once the samples have been prepared and written to :code:`.txt` files, users can use the :code:`train_fasttext` script that reads in the samples within the :code:`.txt` files
in order to train a quality classifier. :code:`train_fasttext` will read in all of the samples within the :code:`.txt` files, split the data into training and
validation sets and train the binary skip-gram classifier. After training, it evaluates the model on the validation samples and writes the predictions
to a jsonl file prints the confusion matrix to stdout.

.. code-block:: bash

    train_fasttext \
      --fasttext-files-dir=${res_dir} \
      --output-train-file=${res_dir}/fasttext_samples.train \
      --output-validation-file=${res_dir}/fasttext_samples.valid \
      --output-model=${res_dir}/cc_filter_test.bin \
      --output-predictions=${res_dir}/preds.jsonl

Finally, with the model trained and able to provide quality scores, it can be used to for quality filtering. Similar to how
:code:`filter_documents` performs language identification with the fastText model :code:`lid.176.bin`, we provide a default config that can
be used for classifier-based quality filtering with a fastText model. Additionally, this filter implements Pareto-based sampling approach
as is described in `Brown et al., 2020 <https://arxiv.org/abs/2005.14165>`_.

.. code-block:: bash

    filter_documents \
      --input-data-dir=<Specify the path to common-crawl/uncurated data> \
      --filter-config-file=./config/fasttext_quality_filter.yaml \
      --output-retained-document-dir=<Output directory to which high-quality documents will be written> \
      --output-removed-document-dir=<Output directory to which low-quality documents will be written> \
      --log-dir=${log_dir}/fasttext_classifier \

-----------------------------------------
Heuristic Filtering
-----------------------------------------

As with other filtering steps, the heuristic-based filtering in NeMo Curator can be carried out using ``ScoreFilter`` or the :code:`filter_documents`
utility. Filters can be chained in NeMo Curator using ``Sequential`` as follows.

.. code-block:: python

    filter_step = nc.Sequential([
        ScoreFilter(
            WordCountFilter(min_words=80),
            score_field="word_count",
        ),
        ScoreFilter(IncompleteStoryFilter()),
        ScoreFilter(RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2)),
        ScoreFilter(RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18)),
        ScoreFilter(RepeatingTopNGramsFilter(n=4, max_repeating_ngram_ratio=0.16)),
    ])

The filter config file :code:`config/heuristic_filter.yaml` provides a generic list of heuristic filters that have been tested
and shown to provide documents that when used for training, lead to improvements in language model downstream task performance.
The filters are general enough that users should feel free to remove certain filters within the cascade of filters and experiment
with the results of different filter configurations/parameters.

Additionally, these filters have been used for curating high-quality non-English documents. However, it is advised that when applying
to non-English data that users write out the document scores by specifying the :code:`--document-score-dir` argument. This will allow users to
examine if a particular filter is responsible for undesirably removing many documents from a corpus.

.. code-block:: bash

    filter_documents \
      --input-data-dir=<Specify path to input dataset> \
      --filter-config-file=./config/heuristic_filter_en.yaml \
      --output-retained-document-dir=<Output directory to which high-quality documents will be written> \
      --output-removed-document-dir=<Output directory to which low-quality documents will be written> \
      --output-document-score-dir=<Output directory to which document scores will be written> \
      --log-dir=${log_dir}/heuristic_filter
