#!/usr/bin/env python3
"""
Setup script for Ingest - Document Processing CLI for RAG
Creates a standalone 'ingest' command
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="ingest-cli",
    version="1.0.2",
    description="High-quality document processing for RAG pipelines, supporting multiple formats and processing backends",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RealTimeX",
    author_email="trung@realtimex.ai",
    url="https://github.com/therealtimex/ingest",
    py_modules=["ingest", "document_cleaner"],
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "ingest=ingest:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Markup :: Markdown",
        "License :: MIT",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="pdf document-processing ocr nlp rag ingest llm table-extraction vector-database",
)
