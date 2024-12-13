import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import cudf
import numpy as np
import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from dask.distributed import get_worker
from nltk.tokenize import sent_tokenize
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from nemo_curator.classifiers.base import DistributedDataClassifier
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, load_object_on_worker
from nemo_curator.utils.script_utils import ArgumentHelper

try:
    from IndicTransTokenizer import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransTokenizer not found. Please install it using the following command: \n"
        + "pip install git+https://github.com/VarunGumma/IndicTransTokenizer.git"
    )

TERMINAL_PUNCTUATIONS = (
    ".",
    "!",
    "?",
    ":",
    ",",
    ";",
    ")",
    "}",
    "]",
    '"',
    "'",
    "”",
    "’",
)
START_PUNCTUATIONS = ("(", "{", "[", "'", '"', "“", "‘")


@dataclass
class TranslationConfig:
    pretrained_model_name_or_path: str
    max_length: int = 50
    num_beams: int = 5
    autocast: bool = False
    max_words_per_sen: int = 200


class CustomModel(nn.Module):
    def __init__(self, config: TranslationConfig):
        super().__init__()
        self.config = config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        self.autocast = config.autocast

    @torch.no_grad()
    def _forward(self, batch: dict) -> torch.Tensor:
        return self.model.generate(
            **batch,
            use_cache=True,
            min_length=0,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            num_return_sequences=1,
            repetition_penalty=1.2,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                outputs = self._forward(batch)
        else:
            outputs = self._forward(batch)
        return outputs


class ModelForSeq2SeqModel(HFModel):
    def __init__(self, config: TranslationConfig):
        self.trans_config = config
        self.config = self.load_config()
        super().__init__(self.trans_config.pretrained_model_name_or_path)

    def load_model(self, device: str = "cuda") -> CustomModel:
        model = CustomModel(
            self.trans_config,
        )
        model = model.to(device)
        model.eval()
        return model

    def load_config(self) -> AutoConfig:
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    @lru_cache(maxsize=1)
    def load_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    def max_seq_length(self) -> int:
        return self.config.max_source_positions

    def load_cfg(self):
        return self.load_config()


class IndicTranslation(DistributedDataClassifier):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "ai4bharat/indictrans2-en-indic-1B",
        input_column: str = "indic_proc_text",
        batch_size: int = 128,
        autocast: bool = False,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.input_column = input_column
        self.batch_size = batch_size
        self.autocast = autocast

        self.translation_config = TranslationConfig(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            max_length=256,
            num_beams=5,
            autocast=self.autocast,
        )
        self.model = ModelForSeq2SeqModel(self.translation_config)
        super().__init__(
            model=self.model,
            batch_size=self.batch_size,
            device_type="cuda",
            autocast=self.autocast,
            labels=None,
            filter_by=None,
            out_dim=None,
            pred_column=None,
            max_chars=None,
        )

    def preprocess_df(self, df: cudf.DataFrame) -> cudf.DataFrame:
        ip = load_object_on_worker(
            "IndicProcessor", IndicProcessor, {"inference": True}
        )
        indices = df["text"].index.to_arrow().to_pylist()
        sentences = df["text"].to_arrow().to_pylist()
        sentences = ip.preprocess_batch(
            sentences, src_lang="eng_Latn", tgt_lang="hin_Deva"
        )
        df["indic_proc_text"] = cudf.Series(sentences, index=indices)
        return df

    def translate_tokens(self, df: cudf.DataFrame) -> cudf.DataFrame:
        worker = get_worker()
        if hasattr(worker, "IndicProcessor"):
            ip = getattr(worker, "IndicProcessor")
        else:
            ip = load_object_on_worker(
                "IndicProcessor", IndicProcessor, {"inference": True}
            )
        tokenizer = self.model.load_tokenizer()
        indices = df["translation"].index.to_arrow().to_pylist()
        generated_tokens = df["translation"].to_arrow().to_pylist()
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )
        generated_tokens = ip.postprocess_batch(generated_tokens, lang="hin_Deva")
        df["translation"] = cudf.Series(data=generated_tokens, index=indices)
        return df

    def has_alphabet_characters(self, text: str) -> bool:
        return any(c.isalpha() for c in text)

    def custom_tokenize(self, text: str):
        split_text = re.split(
            r"(\#{2,}|\_{2,}|\…{2,}|\+{2,}|\.{2,}|\-{3,}|\*{2,}|\~{2,}|\={2,}|\!{2,}|\n|\t|\‣|\⁃|\⁌|\⁍|\●|\○|\•|\·|\◘|\◦|\⦾|\⦿|\|)",
            text,
        )
        split_text = [s for s in split_text if len(s) > 0]
        tokenized_sentences = []
        len_flag = False
        for line in split_text:
            # Tokenize sentences using NLTK's sent_tokenize function
            if self.has_alphabet_characters(line) == True:
                sentences = sent_tokenize(line)
                i = 0
                j = 0
                curr_tokenized_snt = []
                non_translation_str = ""
                # Comparing the list of tokenized sentences (using NLTK) and actual sentence and preserving the spaces,
                # newline and other special characters
                while i < len(line):
                    if j < len(sentences):
                        stripped_sent = sentences[j].strip()
                        if len(stripped_sent) == 0:
                            j += 1
                            continue
                        # If tokenized sentence matches then moving to next sentence
                        if line[i] == stripped_sent[0]:
                            if non_translation_str != "":
                                curr_tokenized_snt.append(non_translation_str)
                            curr_tokenized_snt.append(stripped_sent)
                            i += len(stripped_sent)
                            j += 1
                            non_translation_str = ""
                        else:
                            non_translation_str += line[i]
                            i += 1
                    else:
                        non_translation_str += line[i]
                        i += 1
                if non_translation_str != "":
                    curr_tokenized_snt.append(non_translation_str)
                # Add the tokenized sentences to the list
                tokenized_sentences.extend(curr_tokenized_snt)
            else:
                tokenized_sentences.append(line)

        tokenized_sentence_len = []
        for sentence in tokenized_sentences:
            sent = sentence.split()
            # removing the sentences with word length greater than threshold as the model may not be able translate it due to constraint on output token size
            if len(sent) <= self.translation_config.max_words_per_sen:
                tokenized_sentence_len.append(sentence)

        return tokenized_sentence_len

    def process_input_text(self, df: cudf.DataFrame) -> cudf.DataFrame:
        df = df.to_pandas()
        df["text"] = df["text"].apply(self.custom_tokenize)
        df["doc_id"] = np.arange(1, len(df) + 1)
        df = df.explode("text", ignore_index=True)
        df = df.reset_index(drop=False)
        df = cudf.DataFrame.from_pandas(df)
        return df

    def combine_text(self, df: cudf.DataFrame) -> cudf.DataFrame:
        engligh_stop_flag = df["text"].str.endswith(".")
        hindi_stop_flag = df["translation"].str.endswith("|")
        df["translation"][~engligh_stop_flag & hindi_stop_flag] = df[
            "translation"
        ].str.rstrip("|")
        df["translation"] = df["translation"].str.strip()
        return df

    def grouping(self, df: cudf.DataFrame) -> cudf.DataFrame:
        df = df.to_pandas()
        agg_funcs = {
            "translation": lambda s: "".join(s),
            "text": lambda s: "".join(s),
        }
        other_columns = {
            col: "first"
            for col in df.columns
            if col not in agg_funcs and col != "doc_id"
        }

        agg_funcs.update(other_columns)
        df = df.groupby("doc_id").agg(agg_funcs).reset_index()
        df = cudf.DataFrame.from_pandas(df)
        return df

    def atleast_letter(self, df: cudf.DataFrame, column_name: str) -> cudf.DataFrame:
        df = df.to_pandas()
        df["isalpha"] = df[column_name].apply(self.has_alphabet_characters)
        df = cudf.DataFrame(df)
        return df

    def _run_classifier(self, dataset: DocumentDataset) -> DocumentDataset:
        ddf = dataset.df
        # Applying process_input_text for following :
        # 1. nltk tokenization to break doc into sentences
        # 2. craeting a row w.r.t each sentence.
        # 3. Process sentences strip symbols from start and end
        ddf = ddf.map_partitions(self.process_input_text, enforce_metadata=False)
        ddf["text"] = ddf["text"].astype("str")

        ddf["word_count"] = ddf["text"].str.split().list.len()
        ddf["word_count"] = ddf["word_count"].astype("int64")
        ddf_true = ddf[(ddf["word_count"] <= self.translation_config.max_words_per_sen)]
        # To filter for atleast one unicode letter in text
        has_letter = ddf_true.map_partitions(self.atleast_letter, column_name="text")
        ddf_trans = ddf_true[has_letter["isalpha"]]
        ddf = ddf_trans.drop(columns="word_count")
        ## ddf false operations
        ddf_false = ddf_true[~has_letter["isalpha"]]
        ddf_false = ddf_false.drop(columns="word_count")
        ddf_false["translation"] = ddf_false["text"]
        # Applying preprocess_df for Indic preprocessing
        ddf["text"] = ddf["text"].astype("str")
        ddf_meta = ddf._meta.copy()
        ddf_meta["indic_proc_text"] = ""
        ddf = ddf.map_partitions(self.preprocess_df, meta=ddf_meta)

        columns = ddf.columns.tolist()
        pipe = op.Sequential(
            op.Tokenizer(
                self.model, cols=[self.input_column], tokenizer_type="default"
            ),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                pred_output_col="translation",
            ),
            keep_cols=columns,
        )
        ddf = pipe(ddf)
        translated_meta = ddf._meta.copy()
        translated_meta["translation"] = "DUMMY_STRING"
        ddf = ddf.map_partitions(self.translate_tokens, meta=translated_meta)
        ddf = ddf.map_partitions(self.combine_text, meta=translated_meta)

        # Merging translated and non-translated samples
        ddf_true["false_translation"] = ddf_false["translation"]
        ddf_true["false_translation"] = ddf_true["false_translation"].fillna("")
        ddf_true["translation"] = ddf["translation"]
        ddf_true["translation"] = ddf_true["translation"].fillna("")
        ddf_true["translation"] = (
            ddf_true["translation"] + ddf_true["false_translation"]
        )

        ddf = ddf_true.map_partitions(self.grouping)
        return DocumentDataset(ddf)


def attach_args():
    parser = ArgumentHelper.parse_distributed_classifier_args()
    parser.set_defaults(
        pretrained_model_name_or_path="ai4bharat/indictrans2-en-indic-1B"
    )
    parser.set_defaults(input_text_field="text")
    parser.set_defaults(device="gpu")
    return parser


def main(args):
    print(f"Arguments parsed = {args}")
    st = time.time()
    client = get_client(**ArgumentHelper.parse_client_args(args))
    print(client.dashboard_link)
    translator_model = IndicTranslation(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        input_column=args.input_text_field,
        batch_size=args.batch_size,
        autocast=args.autocast,
    )
    input_files = [
        os.path.join(args.input_data_dir, x) for x in os.listdir(args.input_data_dir)
    ]
    input_dataset = DocumentDataset.read_json(
        input_files, backend="cudf", add_filename=True
    )
    result_dataset = translator_model(dataset=input_dataset)

    result_dataset.to_json(output_path=args.output_data_dir, write_to_filename=True)
    print(f"Total time taken for translation: {time.time()-st} seconds", flush=True)
    client.close()


if __name__ == "__main__":
    main(attach_args().parse_args())
