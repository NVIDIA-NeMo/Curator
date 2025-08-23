import pytest

def test_custom_pii_detection_and_deidentification():
    try:
        import pandas as pd
        from nemo_curator.datasets import DocumentDataset
        from nemo_curator.modifiers.pii_modifier import PiiModifier
        from nemo_curator.modules.modify import Modify
        from nemo_curator.utils.distributed_utils import get_client
        from nemo_curator.pii.custom_recognizers_sample import crypto_recognizer, medical_license_recognizer, iban_generic_recognizer

        client = get_client(cluster_type="gpu")
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
        dataset = DocumentDataset.from_pandas(dataframe, npartitions=1)

        # Recommended entities
        supported_entities=[
            "CRYPTO",
            "MEDICAL_LICENSE",
            "IBAN_CODE",
        ]

        modifier = PiiModifier(
            log_dir="./logs",
            batch_size=8,
            supported_entities=supported_entities,
            anonymize_action="replace",
            custom_analyzer_recognizers=[crypto_recognizer, medical_license_recognizer, iban_generic_recognizer],
        )
        modify = Modify(modifier)
        modified_dataset = modify(dataset)
        datasets = modified_dataset.to_pandas()
        for i, status in enumerate(datasets["text"] == [ "My crypto wallet is <CRYPTO>", "My IBAN is <IBAN_CODE>", "My medical license number is <MEDICAL_LICENSE>"]):
            if status:
                print("De-identification successful:", datasets["text"][i])
                assert True
            else:
                print("De-identification failed:", datasets["text"][i])
                assert False, "Test Failed custom_pii_and_deidentify, data is not de-identified"
    except Exception as e:
        print(e)
        assert False, "Test Failed custom_pii_and_deidentify, code error"