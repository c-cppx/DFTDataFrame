# DFTDataFrames

This repository contains a Python class that can be used to convert a tree of directories with VASP calculations to a Pandas DataFrame object and Tools to interact. Most functions are based on the
Atoms Simulation Environment object Atoms. Have a look at a graphic visualization of this repo with [octo](https://mango-dune-07a8b7110.1.azurestaticapps.net/?repo=c-cppx%2FDFTDataFrame)

## Getting Started

To get started with using this Python class, follow the steps below:

### Prerequisites

Before downloading this repository, ensure that you have Python 3 installed on your machine.

### Downloading the repository

To download this repository, open a terminal window and run the following command:

```python
git clone https://github.com/c-cppx/DFTDataFrame.git
```

### Creat and activate a virtual environment and install the DFTDataFrame package in the environment.

```bash
git clone https://github.com/c-cppx/DFTDataFrame.git
cd DFTDataFrame
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install ./

```

## Usage

To use the Python class in this repository, you can import it into your Python code using the following syntax:

```python
python Readingingoutputfiles.py -r ~/Calculations/ -o DataBase
```

The calculations in folder "Calculations" are now in the excel file "DataBase.xlsx"

I recommend to watch some of [Matt Harrisons tutorials](https://www.youtube.com/results?search_query=matt+harrison+effective+pandas) tutorials about Pandas or read his book ["Effective Pandas"](https://store.metasnake.com/effective-pandas-book).

## Contributing

If you find a bug or would like to suggest a new feature for this repository, please create a new issue or pull request on GitHub.

## License

This project is licensed under the GNU license. See the [LICENSE](LICENSE) file for details.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
