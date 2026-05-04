"""
Setup script for MemBench package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name='membench',
    version='1.0.0',
    author='Anonymous Author(s)',
    author_email='anonymous@example.com',
    description='Comprehensive Evaluation Framework for LLM Memory Systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://anonymous.4open.science/r/membench',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'sentence-transformers>=2.2.0',
        'click>=8.0.0',
        'rank-bm25>=0.2.2',
        'scikit-learn>=1.0.0',
        'tqdm>=4.62.0',
    ],
    extras_require={
        'faiss': ['faiss-cpu>=1.7.0'],
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'viz': [
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'membench=membench.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        'Source': 'https://anonymous.4open.science/r/membench',
    },
)
