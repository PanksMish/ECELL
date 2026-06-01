"""
EOSNM Package Setup Configuration
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="eosnm",
    version="1.0.0",
    author="Pankaj Mishra, V Venkataramanan, Anand Nayyar",
    author_email="pankaj.mishra@somaiya.edu",
    description="Employability Optimization through Skill-Network Modelling",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/PanksMish/ECELL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eosnm=experiment_runner:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "data/*.csv"],
    },
    keywords=[
        "employability",
        "optimization",
        "entrepreneurship education",
        "skill development",
        "network analysis",
        "resource allocation",
        "educational data mining",
        "decision support",
    ],
    project_urls={
        "Bug Reports": "https://github.com/PanksMish/ECELL/issues",
        "Source": "https://github.com/PanksMish/ECELL",
        "Documentation": "https://github.com/PanksMish/ECELL/wiki",
    },
)
