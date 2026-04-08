from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import (
    DistillStage,
    DiverseQAStage,
    ExtractKnowledgeStage,
    KnowledgeListStage,
    WikipediaParaphrasingStage,
)


def test_nemotron_cc_stages_expose_distinct_names() -> None:
    assert WikipediaParaphrasingStage().name == "WikipediaParaphrasing"
    assert DiverseQAStage().name == "DiverseQA"
    assert DistillStage().name == "Distill"
    assert ExtractKnowledgeStage().name == "ExtractKnowledge"
    assert KnowledgeListStage().name == "KnowledgeList"
