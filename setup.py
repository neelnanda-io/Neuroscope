from setuptools import setup

setup(
    name="neuroscope",
    version="0.1.0",
    license="LICENSE",
    description="Code accompanying neuroscope.io, a website for displaying max activating dataset examples for language model neurons",
    install_requires=[
        "einops",
        "numpy",
        "torch",
        "datasets",
        "transformers",
        "tqdm",
        "pandas",
        "datasets",
        "wandb",
        "fancy_einsum",
        "transformer_lens"
    ],
    packages=["neuroscope"],
)
