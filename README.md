# DFTDataFrames

This repository contains a Python class that can be used to convert a tree of directories with VASP calculations to a Pandas DataFrame object and Tools to interact. Most functions are based on the
Atoms Simulation Environment object Atoms.

## Getting Started

To get started with using this Python class, follow the steps below:

### Prerequisites

Before downloading this repository, ensure that you have Python 3 installed on your machine.

### Downloading the repository

To download this repository, open a terminal window and run the following command:

```python
git clone https://github.com/c-cppx/DFTDataFrame.git
```

### Installing dependencies

After downloading the repository, navigate to the root directory of the project and run the following command to install the required Python packages:

```python
pip install -r requirements.txt
```

### Setting up PYTHONPATH

To use the Python class in this repository, you need to add the root directory of this repository to your `PYTHONPATH`. You can do this by running the following command:

```python
export PYTHONPATH=$PYTHONPATH:/path/to/DFTDataFrame
```

Replace `/path/to/DFTDataFrame` with the path to the root directory of this repository on your machine.

## Usage

To use the Python class in this repository, you can import it into your Python code using the following syntax:

```python
from DFTDataFrame import GeometryOptimizationDataFrame

ProjectX = GeometryOptimizationDataFrame('root/to/Calculations')
```

## Contributing

If you find a bug or would like to suggest a new feature for this repository, please create a new issue or pull request on GitHub.

## License

This project is licensed under the GNU license. See the [LICENSE](LICENSE) file for details.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
