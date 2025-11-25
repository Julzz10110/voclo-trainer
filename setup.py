"""Setup script for voclo-trainer."""

from setuptools import setup, find_packages

setup(
    name="voclo-trainer",
    version="0.1.0",
    description="Training system for voclo voice conversion models",
    author="Voclo Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "onnxsim>=0.4.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
)

