============================
Filters
============================

------------------------------
Base Class
------------------------------

.. autoclass:: nemo_curator.filters.DocumentFilter
    :members:
    :member-order: bysource

.. autofunction:: nemo_curator.filters.import_filter

------------------------------
Modules
------------------------------

.. autoclass:: nemo_curator.ScoreFilter
    :members:
    :special-members: __init__, __call__

.. autoclass:: nemo_curator.Score
    :members:
    :special-members: __init__, __call__

.. autoclass:: nemo_curator.Filter
    :members:
    :special-members: __init__, __call__

------------------------------
FastText Filters
------------------------------

.. autoclass:: nemo_curator.filters.FastTextLangId
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.FastTextQualityFilter
    :members:
    :member-order: bysource

------------------------------
Heuristic Filters
------------------------------

.. autoclass:: nemo_curator.filters.NonAlphaNumericFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.SymbolsToWordsFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.NumbersFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.UrlsFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.BulletsFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.WhiteSpaceFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.ParenthesesFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.LongWordFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.WordCountFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.BoilerPlateStringFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.MeanWordLengthFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.RepeatedLinesFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.RepeatedParagraphsFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.RepeatedLinesByCharFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.RepeatedParagraphsByCharFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.RepeatingTopNGramsFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.RepeatingDuplicateNGramsFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.PunctuationFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.EllipsisFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.CommonEnglishWordsFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.WordsWithoutAlphabetsFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.PornographicUrlsFilter
    :members:
    :member-order: bysource

------------------------------
Code Filters
------------------------------

.. autoclass:: nemo_curator.filters.PythonCommentToCodeFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.GeneralCommentToCodeFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.NumberOfLinesOfCodeFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.TokenizerFertilityFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.XMLHeaderFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.AlphaFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.HTMLBoilerplateFilter
    :members:
    :member-order: bysource

.. autoclass:: nemo_curator.filters.PerExtensionFilter
    :members:
    :member-order: bysource