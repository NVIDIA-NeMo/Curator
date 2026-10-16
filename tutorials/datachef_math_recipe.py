import logging
from nemo_curator.recipe import DataChefRecipeGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing DataChefRecipeGenerator...")
    generator = DataChefRecipeGenerator(
        target_benchmark="AIME_25",
        base_model_id="Qwen3-1.7B",
        available_data_sources=["finemath-v1", "open-web-math"],
        compute_budget_tokens=1_000_000,
        api_url="http://localhost:8000/v1/generate"
    )

    logger.info("Generating NeMo Curator pipeline specification using DataChef...")
    yaml_config = generator.generate()
    
    logger.info("Generated Pipeline Configuration:")
    print(yaml_config)

if __name__ == "__main__":
    main()
