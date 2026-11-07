from setuptools import setup, find_packages

setup(
    name="fastmss",
    version="0.1.0",
    description="Fast Multi-Speaker Speech simulation framework",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "soundfile",
        "torch",
        "torchaudio",
        "lhotse",
        "pyroomacoustics",
        "hydra-core",
        "omegaconf",
        "tqdm",
        "urllib3"
    ],
)
