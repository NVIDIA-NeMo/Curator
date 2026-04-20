#!/usr/bin/env python3
"""Download a WeSpeaker pretrained model to a local cache directory.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model w2vbert2_mfa --model-dir ./models
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Download a WeSpeaker pretrained model")
    p.add_argument("--model", default="voxblink2_samresnet100_ft",
                    help="WeSpeaker hub model name")
    p.add_argument("--model-dir", default="models/",
                    help="Local directory to cache the model")
    args = p.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.environ["WESPEAKER_HOME"] = os.path.abspath(args.model_dir)

    import wespeaker

    logger.info("Downloading model: %s", args.model)
    model = wespeaker.load_model(args.model)

    total_params = sum(p.numel() for p in model.model.parameters())
    logger.info("Total parameters: %s", f"{total_params:,}")

    if hasattr(model.model, "frontend") and model.model.frontend is not None:
        fe_params = sum(p.numel() for p in model.model.frontend.parameters())
        logger.info("Frontend parameters: %s", f"{fe_params:,}")

    logger.info("Model saved to: %s/%s/", args.model_dir, args.model)


if __name__ == "__main__":
    main()
