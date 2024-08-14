
.. _data-curator-download:

======================================
Download and Extract Text
======================================
-----------------------------------------
Background
-----------------------------------------
Publicly hosted text datasets are stored in various locations and formats. Downloading a massive public dataset is usually the first step in data curation,
and it can be cumbersome due to the dataset's massive size and hosting method.
Also, massive pretraining text datasets are rarely in a format that can be immediately operated on for further curation and training.
For example, the Common Crawl stores its data in a compressed web archive format (:code:`.warc.gz`) for its raw crawl data, but formats
like :code:`.jsonl` are more common for data curation due to their ease of use.
However, extraction can be by far the most computational expensive step of the data curation pipeline, so it can be beneifical to do some filtering prior to
the extraction step to limit the amount of documents that undergo this heavy computation.

NeMo Curator provides example utilities for downloading and extracting Common Crawl, ArXiv, and Wikipedia data.
In addition, it provides a flexible interface to extend the utility to other datasets.
Our Common Crawl example demonstrates how to process a crawl by downloading the data from S3, doing preliminary language filtering with pyCLD2,
and extracting the relevant text with jusText or Resiliparse to output :code:`.jsonl` files.

NeMo Curator currently does not provide out-of-the-box support for web-crawling or web-scraping.
It provides utilities for downloading and extracting data from the preexisting online sources given above.
Users can easily implement these functions themselves and automatically scale them with the framework described below if they would like.

-----------------------------------------
Usage
-----------------------------------------

``nemo_curator.download`` has a collection of functions for handling the download and extraction of online datasets.
By "download", we typically mean the transfer of data from a web-hosted data source to local file storage.
By "extraction", we typically mean the process of converting a data format from its raw form (e.g., ``.warc.gz``) to a standardized format (e.g., ``.jsonl``) and discarding irrelvant data.

* ``download_common_crawl`` will download and extract the compressed web archive files of common crawl snapshots to a target directory.
  Common crawl has an S3 bucket and a direct HTTPS endpoint. If you want to use the S3 bucket, ensure you have properly set up your credentials with `s5cmd <https://github.com/peak/s5cmd>`_.
  Otherwise, the HTTPS endpoints will be used with ``wget``. Here is a small example of how to use it:

  .. code-block:: python

    from nemo_curator.download import download_common_crawl

    common_crawl = download_common_crawl("/extracted/output/folder", "2020-50", "2021-04", output_type="jsonl")

  * ``"/extracted/output/folder"`` is the path to on your local filesystem where the final extracted files will be placed.
  * ``"2020-50"`` is the first common crawl snapshot that will be included in the download. **Note:** Not every year and week has a snapshot. Ensure that your range includes at least one valid Common Crawl snapshot. A list of valid Common Crawl snapshots can be found `here <https://data.commoncrawl.org/>`_.
  * ``"2021-04"`` is the last common crawl snapshot that will be included in the download.
  * ``output_type="jsonl"`` is the file format that will be used for storing the data on disk. Currently ``"jsonl"`` and ``"parquet"`` are supported.

You can choose to modify the HTML text extraction algorithm used in ``download_common_crawl``. See an example below.

  .. code-block:: python

    from nemo_curator.download import (
      ResiliparseExtractor,
      download_common_crawl,
    )

    # Change the extraction algorithm
    extraction_algorithm = ResiliparseExtractor()
    common_crawl = download_common_crawl(
      "/extracted/output/folder",
      "2020-50",
      "2021-04",
      output_type="jsonl",
      algorithm=extraction_algorithm,
    )

  Above, we changed the extraction algorithm from the default ``JusTextExtractor``.

  The return value ``common_crawl`` will be in NeMo Curator's standard ``DocumentDataset`` format. Check out the function's docstring for more parameters you can use.

  NeMo Curator's Common Crawl extraction process looks like this under the hood:

 1. Decode the HTML within the record from binary to text.
 2. If the HTML can be properly decoded, then with `pyCLD2 <https://github.com/aboSamoor/pycld2>`_, perform language detection on the input HTML.
 3. Finally, the extract the relevant text with `jusText <https://github.com/miso-belica/jusText>`_ or `Resiliparse <https://github.com/chatnoir-eu/chatnoir-resiliparse>`_ from the HTML and write it out as a single string within the 'text' field of a json entry within a `.jsonl` file.
* ``download_wikipedia`` will download and extract the latest wikipedia dump. Files are downloaded using ``wget``. Wikipedia might download slower than the other datasets. This is because they limit the number of downloads that can occur per-ip address.

  .. code-block:: python

    from nemo_curator.download import download_wikipedia

    wikipedia = download_wikipedia("/extracted/output/folder", dump_date="20240201")

  * ``"/extracted/output/folder"`` is the path to on your local filesystem where the final extracted files will be placed.
  * ``dump_date="20240201"`` fixes the Wikipedia dump to a specific date. If no date is specified, the latest dump is used.

* ``download_arxiv`` will download and extract latex versions of ArXiv papers. They are hosted on S3, so ensure you have properly set up your credentials with `s5cmd <https://github.com/peak/s5cmd>`_.

  .. code-block:: python

    from nemo_curator.download import download_arxiv

    arxiv = download_arxiv("/extracted/output/folder")

  * ``"/extracted/output/folder"`` is the path to on your local filesystem where the final extracted files will be placed.


All of these functions return a ``DocumentDataset`` of the underlying dataset and metadata that was obtained during extraction. If the dataset has been downloaded and extracted at the path passed to it, it will read from the files there instead of downloading and extracting them again.
Due to how massive each of these datasets are (with Common Crawl snapshots being on the order of hundreds of terrabytes) all of these datasets are sharded accross different files.
They all have a ``url_limit`` parameter that allows you to only download a small number of shards.

-----------------------------------------
Related Scripts
-----------------------------------------
In addition to the Python module described above, NeMo Curator provides several CLI scripts that you may find useful for performing the same function.

The :code:`download_and_extract` script within NeMo Curator is a generic tool that can be used to download and extract from a number of different
datasets. In general, it can be called as follows in order to download and extract text from the web:

.. code-block:: bash

  download_and_extract \
    --input-url-file=<Path to .txt file containing list of URLs> \
    --builder-config-file=<Path to .yaml file that describes how the data should be downloaded and extracted> \
    --output-json-dir=<Path to output directory to which data will be written in .jsonl format>

This utility takes as input a list of URLs that point to files that contain prepared, unextracted data (e.g., pre-crawled web pages from Common Crawl), a config file that describes how to download and extract the data, and the output directory where the extracted text will be written in jsonl format (one json written to each document per line). For each URL provided in the list of URLs, a corresponding jsonl file will be written to the output directory.

The config file that must be provided at runtime, should take the following form:

.. code-block:: yaml

  download_module: nemo_curator.download.mydataset.DatasetDownloader
  download_params: {}
  iterator_module: nemo_curator.download.mydataset.DatasetIterator
  iterator_params: {}
  extract_module: nemo_curator.download.mydataset.DatasetExtractor
  extract_params: {}

Each pair of lines corresponds to an implementation of the abstract DocumentDownloader, DocumentIterator and DocumentExtractor classes. In this case the dummy names of DatasetDownloader, DatasetIterator, and DatasetExtractor have been provided. For this example, each of these have been defined within the fictitious file :code:`nemo_curator/download/mydataset.py`. Already within NeMo Curator, we provide implementations of each of these classes for the Common Crawl, Wikipedia and ArXiv datasets.

###############################
Common Crawl Example
###############################


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Set Up Common Crawl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you prefer, the download process can pull WARC files from S3 using `s5cmd <https://github.com/peak/s5cmd>`_.
This utility is preinstalled in the NeMo Framework Container, but you must have the necessary credentials within :code:`~/.aws/config` in order to use it.
If you prefer to use this method instead of `wget <https://en.wikipedia.org/wiki/Wget>`_ , set :code:`aws=True` in the :code:`download_params` as follows:

.. code-block:: yaml

  download_module: nemo_curator.download.commoncrawl.CommonCrawlWARCDownloader
  download_params:
    aws: True
  iterator_module: nemo_curator.download.commoncrawl.CommonCrawlWARCIterator
  iterator_params: {}
  extract_module: nemo_curator.download.commoncrawl.CommonCrawlWARCExtractor
  extract_params: {}


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Download and Extract Common Crawl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As described in the first section of this document, the first step in using the :code:`download_and_extract` for Common Crawl is to create a list of URLs that point to the location of the WARC files hosted by Common Crawl.
Within NeMo Curator, we provide the :code:`get_common_crawl_urls` utility to obtain these URLs. This utility can be run as follows:

.. code-block:: bash

  get_common_crawl_urls \
    --cc-snapshot-index-file=./url_data/collinfo.json \
    --starting-snapshot="2020-50" \
    --ending-snapshot="2020-50" \
    --output-warc-url-file=./url_data/warc_urls_cc_2020_50.txt

This script pulls the Common Crawl index from `https://index.commoncrawl.org` and stores the index to the file
specified by the argument :code:`--cc-snapshot-index-file`. It then retrieves all WARC URLs between the
dates specified by the arguments :code:`--starting-snapshot` and :code:`--ending-snapshot`.
Finally, it writes all WARC URLs to the text file :code:`--output-warc-urls`. This file is a simple text file
with the following format::

  https://data.commoncrawl.org/crawl-data/CC-MAIN-2020-50/segments/1606141163411.0/warc/CC-MAIN-20201123153826-20201123183826-00000.warc.gz
  https://data.commoncrawl.org/crawl-data/CC-MAIN-2020-50/segments/1606141163411.0/warc/CC-MAIN-20201123153826-20201123183826-00001.warc.gz
  https://data.commoncrawl.org/crawl-data/CC-MAIN-2020-50/segments/1606141163411.0/warc/CC-MAIN-20201123153826-20201123183826-00002.warc.gz
  https://data.commoncrawl.org/crawl-data/CC-MAIN-2020-50/segments/1606141163411.0/warc/CC-MAIN-20201123153826-20201123183826-00003.warc.gz
  https://data.commoncrawl.org/crawl-data/CC-MAIN-2020-50/segments/1606141163411.0/warc/CC-MAIN-20201123153826-20201123183826-00004.warc.gz
  ...

For the CC-MAIN-2020-50 snapshot, there are a total of 72,000 compressed WARC files each between 800 - 900 MB.

Now with the prepared list of URLs, we can use the Common Crawl config included in the :code:`config` directory under the root directory of the repository. This config uses the download, data loader, and extraction classes defined in the file :code:`nemo_curator/download/commoncrawl.py`.
With this config and the input list of URLs, the :code:`download_and_extract` utility can be used as follows for downloading and extracting text from Common Crawl:

.. code-block:: bash

    download_and_extract \
      --input-url-file=./url_data/warc_urls_cc_2020_50.txt \
      --builder-config-file=./config/cc_warc_builder.yaml \
      --output-json-dir=/datasets/CC-MAIN-2020-50/json


As the text is extracted from the WARC records, the prepared documents are written to the directory specified by :code:`--output-json-dir`. Here is an
example of a single line of an output `.jsonl` file extracted from a WARC record:

.. code-block:: json

   {"text": "커뮤니티\n\n어린이 요리 교실은 평소 조리와 제과 제빵에 관심이 있는 초등학생을 대상으로 나이프스킬, 한식, 중식, 양식, 제과, 제빵, 디저트,
    생활요리 등 요리 기초부터 시작해 다양한 요리에 대해 배우고, 경험할 수 있도록 구성되었다.\n\n요즘 부모들의 자녀 요리 교육에 대한 관심이 높아지고
    있는데, 어린이 요리교실은 자녀들이 어디서 어떻게 요리를 처음 시작할지 막막하고 어려워 고민하는 이들을 위해 만들어졌다.\n\n그 뿐만 아니라 학생들이
    식재료를 다루는 과정에서 손으로 만지고 느끼는 것이 감각을 자극하여 두뇌발달에 도움을 주며, 조리를 통해 자신의 감정을 자연스럽게 표현할 수
    있고 이를 통해 정서적 안정을 얻을 수 있다. 또한, 다양한 사물을 만져 보면서 차이점을 구별하고 사물의 특징에 대해 인지할 수 있으므로 인지 능력 향상에
    도움이 되며, 만지고 느끼고 비교하는 과정에서 감각 기능을 향상시킬 수 있다.\n\n방과 후 시간이 되지 않는 초등학생들을 위해 평일반 뿐만 아니라 주말반도
    운영하고 있으며 두 분의 선생님들의 안전적인 지도하에 수업이 진행된다. 한국조리예술학원은 젊은 감각과 학생들과의 소통을 통해 자발적인 교육을 가르친다.
    자세한 학원 문의는 한국조리예술학원 홈페이지나 대표 전화, 카카오톡 플러스친구를 통해 가능하다.", "id": "a515a7b6-b6ec-4bed-998b-8be2f86f8eac",
    "source_id": "https://data.commoncrawl.org/crawl-data/CC-MAIN-2020-50/segments/1606141163411.0/warc/CC-MAIN-20201123153826-20201123183826-00000.warc.gz",
    "url": "http://hanjowon.co.kr/web/home.php?mid=70&go=pds.list&pds_type=1&start=20&num=67&s_key1=&s_que=", "language": "KOREAN"}

Once all records have been processed within a WARC file, it is by default deleted from disk.
