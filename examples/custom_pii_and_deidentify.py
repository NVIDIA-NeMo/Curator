import pandas as pd
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.pii.custom_recognizers_sample import crypto_recognizer, medical_license_recognizer, iban_generic_recognizer

import argparse
from nemo_curator.utils.script_utils import ArgumentHelper

def main(args: argparse.Namespace) -> None:
    # define client: choice "cpu" or "gpu"
    client = get_client(**ArgumentHelper.parse_client_args(args))

    # create a sample dataframe
    dataframe = pd.DataFrame(
        {
            "text": 
                [
                    "My crypto wallet is 0x32Be343B94f860124dC4fEe278FDCBD38C102D88",
                    "My IBAN is GB33BUKB20201555555555",
                    "My medical license number is MED1234567"
                ]
        }
    )

    # Load data - be DocumentDataset
    dataset = DocumentDataset.from_pandas(dataframe, npartitions=1)

    # Initialize PiiModifier
    modifier = PiiModifier(
        log_dir="./logs",
        batch_size=2,
        supported_entities=["CRYPTO", "MEDICAL_LICENSE", "IBAN_CODE"], # Custom entities (sample)
        anonymize_action="replace",
        custom_analyzer_recognizers=[crypto_recognizer, medical_license_recognizer, iban_generic_recognizer], # Custom recognizers (sample)
    )
    modify = Modify(modifier)
    modified_dataset = modify(dataset)
    datasets = modified_dataset.to_pandas()

    # (Optional) Save the modified dataset
    datasets.to_csv("./modified_data.csv", index=False)

def attach_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return ArgumentHelper(parser).add_distributed_args()

if __name__ == "__main__":
    main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())