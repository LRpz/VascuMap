from setuptools import setup, find_packages

setup(
    name="vascumap",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "tifffile",
        "albumentations",
        "segmentation-models-pytorch",
        "catalyst",
        "scikit-image",
        "ttach",
        "pandas",
        "networkx",
        "sknw",
        "alphashape",
        "shapely",
        "scipy",
        "matplotlib",
        "tqdm",
        "joblib",
    ],
    author="Luca Rappez",
    description="Label-free phenotyping of human microvessel networks",
    python_requires=">=3.8",
)