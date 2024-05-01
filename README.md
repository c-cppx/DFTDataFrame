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

### Actiave the virtual environment 

The repository contains a hidden folder *.venv* which contains a virtual environment that you can use for this package and the Jupyter Notebooks.  To activate it go to the folder and type:
```bash
source .venv/bin/activate
```
You can also choose to not use a virtual environmen for this package.
Then install the local repository with pip either in the virtual environment or the environment of your choice.


```bash
pip install .
```

## Usage

To use the Python class in this repository, you can import it into your Python code using the following syntax:

```python
from DFTDataFrame import Tools

root='Path_to_calculations' # The path to the folder that contains all calculations you want to have in your frame.
flag_file = 'final.traj' # A function will look for all subfolders in the root that contain this file and only include those in the frame.
calc_file='final.traj'  # Those files will be read with ase.read to get the final structure and energy
YourCalculationsFrame = create_frame(root, calc_file='final.traj', flag_file='final.traj')

```
YourCalculationsFrame is now a pandas DataFrame object. Many functions for reading e.g:
- Frequencies, Bader charges from ACF.dat are provided in the Tools file. Example of how to use them are shsown in the the attached Jupyter Notebooks Adsorbates. 
- Inputparameter checks for consistency among the calculations for Kpoints and others are described in InputParameters.ipynb


In the Inputp. I recommend to watch some of [Matt Harrisons tutorials](https://www.youtube.com/results?search_query=matt+harrison+effective+pandas) tutorials about Pandas or read his book ["Effective Pandas"](https://store.metasnake.com/effective-pandas-book).

## Contributing

If you find a bug or would like to suggest a new feature for this repository, please create a new issue or pull request on GitHub.

## License

This project is licensed under the GNU license. See the [LICENSE](LICENSE) file for details.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
