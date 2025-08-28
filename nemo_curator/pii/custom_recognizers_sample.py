from presidio_analyzer import PatternRecognizer, Pattern

crypto_recognizer = PatternRecognizer(
    supported_entity="CRYPTO",
    patterns=[
        Pattern(name="Ethereum wallet", regex="0x[a-fA-F0-9]{40}", score=0.9)
    ]
)

medical_license_recognizer = PatternRecognizer(
    supported_entity="MEDICAL_LICENSE",
    patterns=[
        Pattern(name="Medical license", regex="MED[0-9]{7}", score=0.9)
    ]
)

iban_generic_recognizer = PatternRecognizer(
    supported_entity="IBAN_CODE",
    patterns=[
        Pattern(
            name = "IBAN Code",
            regex = r"\b([A-Z]{2})([0-9]{2})([A-Z]{4})([A-Z0-9]{14})\b",
            score = 0.9,
        )
    ]
)