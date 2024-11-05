# DFTDataFrames

This repository contains a set of Python functions to funnel output from Quantum Chemical Calculation with the help of ASE into a pandas DataFrame. The main intention was to track the adsorption energy for heterogeneous catalysis reactions including reaction barriers and free energy from calculated entropy values with ASE.

The most essential functions like reading input and output files are ASE dependent as it saves the Atoms Object in a columns. csv format like files can be read by pandas itself. 
Atoms Simulation Environment object Atoms. Have a look at a graphic visualization of this repo with [octo](https://mango-dune-07a8b7110.1.azurestaticapps.net/?repo=c-cppx%2FDFTDataFrame)

## Getting Started

To get started with using this Python class, follow the steps below:

### Prerequisites

Python. Ideally Python 3. And some calculations that can be read from the ASE io file 

```bash
git clone https://github.com/c-cppx/DFTDataFrame.git
```

### Probably its better to install the DFTDataFrame package in an virtual environment, but it mostly  uses pandas, ase, sympy packages.

```bash
conda create --name DFTDF python=3.11
conda activate DFTDF
cd DFTDataFrame
pip3 install ./ -r requirements.txt
pip3 install ./
```

## Example
```python
from DFTDataFrame.Tools import crate_frame

root='Path_to_calculations' # The path to the folder that contains all calculations you want to have in your frame.
flag_file = 'final.traj' # A function will look for all subfolders in the root that contain this file and only include those in the frame.
calc_file='final.traj'  # Those files will be read with ase.read to get the final structure and energy
OnePiece = create_frame(root, calc_file='final.traj', flag_file='final.traj')

```
OnePiece is now a pandas DataFrame that contains caclulation under the root (that have a final.traj .) Many functions for reading e.g:
- ASE Harmonic Thermochemistry output for free energy calculations and checking Frequencies. One column containing equations (6.233  - T * 0.000123 = G) with sympy objects.  
- Reading the Bader Chargeds from VASP and Henkelmann groups bader analysis script generated ACF.dat.
- Inputparameter checks for consistency among the calculations for Kpoints and others are described in InputParameters.ipynb
- Example Jupyter Notebooks for
    - creating and expanding a DataFrame from information obtained in the Atoms object or the Path.
    - Reading entropies
    - Writing to an excel file even with integrated formulas. (To double check all calculations from within excel)
  

I recommend to watch some of [Matt Harrisons tutorials](https://www.youtube.com/results?search_query=matt+harrison+effective+pandas) tutorials about Pandas or read his book ["Effective Pandas"](https://store.metasnake.com/effective-pandas-book).  

## Contributing

If you find a bug or would like to suggest a new feature for this repository, please create a new issue or pull request on GitHub or E-Mail me directly.

## License

This project is licensed under the GNU license. See the [LICENSE](LICENSE) file for details.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
