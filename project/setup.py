#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Bridge VIV Risk Assessment System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="bridge-viv-assessment",
    version="1.0.0",
    description="Bridge Vortex-Induced Vibration Risk Assessment System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Bridge VIV Research Team",
    author_email="bridge-viv@example.com",
    url="https://github.com/bridge-viv/assessment-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "jupyter>=1.0.0",
        ],
        "ml": [
            "shap>=0.40.0",
            "mlflow>=1.20.0",
            "wandb>=0.12.0",
        ],
        "web": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
        ],
        "cloud": [
            "boto3>=1.18.0",
            "google-cloud-storage>=1.42.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "bridge-viv-train=train:main",
            "bridge-viv-predict=predict:main",
            "bridge-viv-experiments=experiments:main",
            "bridge-viv-hyperopt=hyperparam_search:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="bridge engineering, vortex-induced vibration, machine learning, risk assessment",
    project_urls={
        "Bug Reports": "https://github.com/bridge-viv/assessment-system/issues",
        "Source": "https://github.com/bridge-viv/assessment-system",
        "Documentation": "https://bridge-viv.readthedocs.io/",
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
    zip_safe=False,
)