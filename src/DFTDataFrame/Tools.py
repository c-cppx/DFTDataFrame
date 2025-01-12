"""Functions for the DataFrame creation."""

import logging
import re
from os import listdir
from os import path as ospath
from os import walk
from os.path import getmtime

import matplotlib.pyplot as plt

import numpy as np

from sympy import Symbol, im, I
from sympy import re as real
from sympy import symbols

from time import ctime
from ase import Atoms
from ase.io import read as aseread
from IPython.display import display
from numpy import NaN
from numpy import min as npmin
from pandas import DataFrame, read_csv
import pandas as pd
from pathlib import Path
from ase.constraints import FixAtoms
from typing import IO
from tqdm import tqdm

from ase.neighborlist import neighbor_list


def get_project_root() -> Path:
    return str(Path(__file__).parent.parent)

def get_pathtofile(root, file):
    filepath = "/".join([root, file]).replace("///", "/").replace("//", "/")
    return filepath

def read_strucfile(row, file=None):
    """Reads file with ase. Checks if the file exists and is readable. Returns Atoms object."""
    if ospath.exists(get_pathtofile(row["Path"], file)):
        try:
            return aseread(get_pathtofile(row["Path"], file))
        except IndexError:
            logging.critical(file + " empty in " + row.Name)
            return Atoms()
        except Exception as error:
            logging.critical(str(error)+' ' + file + ' '  + row.Name)
            return Atoms()
    else:
        return Atoms()


def read_CONTCAR(row):
    """aseread CONTCAR if it exists and is not empty"""
    if ospath.exists(row["Path"] + "/CONTCAR"):
        try:
            return aseread(row["Path"] + "/CONTCAR")
        except IndexError:
            logging.info("exists but empty: %s" % row.Path)
        except Exception as error:
            print(
                row.Name, "An error occured:", type(error).__name__
            )  # An error occured
            logging.info(row.Path + "exists but could not be read")
    else:
        return Atoms()


def getverboseprint(verbose):
    """Print only when verbose is True"""
    if verbose:
        def verboseprint(*args):
            # Print each argument separately
            for arg in args:
                logging.info(arg)
    else:
        def verboseprint(*args):
            pass  # do-nothing function

    return verboseprint


def getfmax(struc):
    try:
        fmax = round(np.max(np.abs(struc.get_forces())), 4)
    except Exception:
        fmax = 1
    return fmax

def crawl(root, flag="out.txt") -> list:
    """Crawl through the subfolders of the given root and return the paths
    that contain the flag file."""
    paths = []

    logging.basicConfig(level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger('crawl')
    
    for path, _folders, files in walk(root, followlinks=True):
        if flag in files:
            paths.append(path)
    if len(paths) == 0:
        logger.critical("No " + flag + " found in ", root)
    else:
        logger.info(f"Found {len(paths)} folders in {root}")

    assert len(paths) > 0, 'No places found containing ' + flag + ' in ' + root

    return paths


def makename(path, root, droplist=None, replacedic=None):
    """Creates the index from the path string"""
    name = path.replace(root, "").replace("/", "-")  # root
    if droplist is not None:
        for string in droplist:
            name = name.replace(string, "")

    if replacedic is not None:
        for string1, string2 in replacedic.items():
            name = name.replace(string1, string2)
    name = name.replace("--", "-")
    return name


def fill_struc_gap(row, fillwith, verbose=False):
    """Fills the gaps in a structure column with the structure from a given file."""
    if verbose:
        logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger('fill_struc_gap')
    else:
        logging.basicConfig(level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger('fill_struc_gap')

    if row.struc == Atoms():
        struc = read_strucfile(row, fillwith)
        try:
            E = struc.get_potential_energy()
            fmax = np.max(struc.get_forces())
        except RuntimeError as error:
            logger.debug(
                row.Name+  " An error occured: "+ type(error).__name__
            )  # An error occured
            E = NaN
            fmax = NaN
        except StopIteration:
            logger.debug(
                row.Name+  " StopIteration error occured: "
            )  # An error occured
            E = NaN
            fmax = NaN
        except Exception as error:
            logger.debug(
                row.Name+  " Unexpected error: "+ type(error).__name__
            )  # An error occured
            E = NaN
            fmax = NaN
        return struc, E, fmax
    else:
        return row.struc, row.E, row.fmax


def read_relaxed_structure(row, calc_file="OUTCAR", verbose=False):
    """Read the relaxed structure from OUTCAR, .traj or other ase compatible
    files.
    Returns the structure as atoms object, cellparameters(a,b,c,gamma)
      and final energy.

    :param calc_file: output file of geometric relaxation calculation,
     defaults to OUTCAR
    :type calc_file: str()
    ...
    :raises [RunTimeError]: The calculation file exists, but does not have
      an energy
    ...
    :return: energy, calc, fmax, human_time, timestamp, cell_a, cell_b, cell_c, gamma,
      formula, constraints
    :rtype: float, Atoms, float, str, float, float, float, float, str, list
    """
    if verbose:
        logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger = logging.getLogger('read_relaxed_structure')
    else:
        logging.basicConfig(level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger = logging.getLogger('read_relaxed_structure')
    path = row["Path"]
    # print([path +'/'+ n for n in os.listdir(path)])
    timestamp = getmtime(path)
    human_time = ctime(timestamp)
    energy = 0
    calc = Atoms()
    fmax = 0
    cell_a = 0
    cell_b = 0
    cell_c = 0
    gamma = 0
    formula = 0
    constraints = []
    pathtofile = get_pathtofile(path, calc_file)
    if ospath.exists(pathtofile):
        try:
            calc = aseread(pathtofile)
        except Exception as error:
            logger.debug(("{0} could not be read "+error).format(calc_file))
            calc = Atoms()
        else:
            try:
                energy = calc.get_potential_energy()
                fmax = round(np.max(np.abs(calc.get_forces())), 4)
            except RuntimeError:
                logging.debug("%s energy could not be read", pathtofile)
                energy = 0
                fmax = 0
            else:
                cell_a = calc.cell.cellpar()[0]
                cell_b = calc.cell.cellpar()[1]
                cell_c = calc.cell.cellpar()[2]
                gamma = calc.cell.cellpar()[5]
                formula = calc.get_chemical_formula()
                constraints = calc.constraints

    else:
        logger.debug("%s does not exist "+ pathtofile)
        calc = Atoms()
    if calc == 0:
        logger.debug("Outstanding error")
    return (
        energy,
        calc,
        fmax,
        timestamp,
        human_time,
        cell_a,
        cell_b,
        cell_c,
        gamma,
        formula,
        constraints
    )


def create_frame(
    root,
    flag_file="out.txt",
    calc_file="OUTCAR",
    droplist=None,
    replacedic=None,
    verbose=False
    ) -> DataFrame:
    """
    - flag_file: The file to look for when walking the folders that contain
        calculations.
    - calc_file: The structure file that contains propiertes from the
        calculation (calc, final.traj). read with ase.io.aseread
    - struc: The final structure after relaxation. aseread with ase.io.aseread
    - droplist: a list of strings that can be ommitted when creating the Name from the Path e.g slab, surface
    """
    if droplist is None:
        droplist = []

    if not root.endswith("/"):
        root = root + "/"

    if verbose:
        logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",)

        logger = logging.getLogger('create_frame')

    paths = crawl(root=root, flag=flag_file)
    assert len(paths) > 0, 'No places found containing ' + flag_file + ' in ' + root

    frame = DataFrame(paths, columns=["Path"])
    frame["Name"] = frame["Path"].apply(makename, args=[root, droplist, replacedic])
    frame = DataFrame(
        frame.Path.to_list(),
        index=frame["Name"].to_list(),
        columns=["Path"],
    )
    frame["Name"] = frame.index
    frame["files"] = frame["Path"].apply(listdir)

    if len(frame) == 0:
        logger.critical("frame is empty")

    else:
        tqdm.pandas(desc="reading structures ")

        (
            frame["E"],
            frame["struc"],
            frame["fmax"],
            frame["timestamp"],
            frame["human_time"],
            frame["a"],
            frame["b"],
            frame["c"],
            frame["gamma"],
            frame["Formula"],
            frame["constraints"]
        ) = zip(*frame.progress_apply(read_relaxed_structure, args=[calc_file, verbose], axis=1))

    return DataFrame(frame)


def update(
    frame,
    root,
    calc_file="OUTCAR",
    flag_file="out.txt",
    verbose=False,
    droplist=None,
):
    """
    Update modified folders. Newly found folders are added.
    Not existing paths are removed from the dataframe.
    """
    if droplist is None:
        droplist = []

    def is_modified(row):
        """Compare timestamp with last modified of calculations folder."""
        try:
            return row.timestamp > getmtime(row.Path)
        except FileNotFoundError:
            logging.info("Path does not exist " + row.Path)
            return False

    def nonexistent_paths(row):
        """Remove from Frame if the path does not exist anymore. Returns boolean series. True if it should be removed."""
        return ospath.exists(row.Path)

    def remove_nonexistent_paths(frame):
        nonexistant = frame[~frame.apply(nonexistent_paths, axis=1)].index
        print(len(nonexistant), "were removed")
        return frame.drop(nonexistant)

    newpaths = crawl(root, flag_file)
    for ind, row in frame.iterrows():
        if row.Path in newpaths:
            newpaths.remove(row.Path)
        else:
            print(ind, " was moved or renamed")
    modifiedpaths = frame[frame.apply(is_modified, axis=1)].Path.to_list()
    print("modified", len(modifiedpaths))
    paths_to_update = modifiedpaths
    if paths_to_update:
        print("update", len(paths_to_update), "folders")
        new = DataFrame(newpaths, columns=["Path"])
        new["Name"] = new["Path"].apply(makename, args=[root, droplist])
        new = DataFrame(
            new.Path.to_list(), index=new["Name"].to_list(), columns=["Path"]
        )
        new["Name"] = new.index
        new["files"] = new["Path"].apply(listdir)
        if verbose:
            display("new", new)
        try:
            (
                new["E"],
                new["struc"],
                new["fmax"],
                new["timestamp"],
                new["human_time"],
                new["a"],
                new["b"],
                new["c"],
                new["gamma"],
                new["Formula"],
            ) = zip(
                *new.apply(
                    read_relaxed_structure,
                    args=[calc_file, verbose],
                    axis=1,
                )
            )
        except Exception as error:
            print(
                row.Name, "An error occured:", type(error).__name__
            )  # An error occured
            for i, n in new.iterrows():
                n = read_relaxed_structure(n, calc_file, verbose)
                if len(n) != 9:
                    print(i, "read_relaxed_eror")
                    print(n)
        if verbose:
            display(new)
        for (
            rem
        ) in (
            new.index.to_list()
        ):  # remove lines from frame which are added with the new data
            if rem in frame.index:
                frame.drop(rem, inplace=True)
        frame = pd.concat([frame, new])
    else:
        print("Nothing to update")

    frame = remove_nonexistent_paths(frame)

    return frame


def group_min(Frame, group, value):
    MLgroup = Frame.groupby(group)
    ML_max = DataFrame(columns=Frame.columns)
    for name, group in MLgroup:
        group = MLgroup.get_group(name)
        ML_max = pd.concat([ML_max, group.sort_values(value).head(n=1)])
    return ML_max

def group_max(Frame, group, value):
    MLgroup = Frame.groupby(group)
    ML_max = DataFrame(columns=Frame.columns)
    for name, group in MLgroup:
        group = MLgroup.get_group(name)
        ML_max = pd.concat([ML_max, group.sort_values(value).tail(n=1)])
    return ML_max

def get_constraints(row):
    constraints =  row.constraints
    if len(constraints) == 0:
        constraints = [FixAtoms(indices=[])]
    return constraints

def get_all_elements(Frame):
    try:
        elements = Frame.apply(lambda x: x['struc'].get_chemical_symbols(), axis=1)
        return set([item for sublist in elements for item in sublist])
    except Exception as error:
        print('error when trying to call get_chemical_symbols()', error)

def count_element(row, element, struc="struc"):
    try:
        traj = row[struc]
    except Exception as error:
        print(row.Name, error)
    count_of_element = len([atom.symbol for atom in traj if atom.symbol == element])
    return count_of_element

def get_element_counts(Frame):
    element_list = get_all_elements(Frame)
    for el in element_list:
        Frame[el] = Frame.apply(count_element, args=[el], axis=1)
    return Frame


def distance_from_surface(
    row, struc=None, adsorbate_atoms=["C", "O", "H"], all_distances=False
):
    """Return the distance of the adsorbate from the distance. Atoms with C, O, H are considered part of the adsorbate.
    Returns all distances or the minimum distance of C, O or H to any other element that is not C, O or H.
    """
    distances = {}
    adsorbates_index = {}
    struc = row[struc]
    if struc is not Atoms():
        indices = struc.symbols.indices()
    for adsorbate_atom in adsorbate_atoms:
        try:
            adsorbates_index[adsorbate_atom] = indices.pop(adsorbate_atom)
        except Exception:
            # print("An error occured:", type(error).__name__) # An error occured
            continue
    for adsorbate_atom in adsorbates_index.keys():
        for el in indices.keys():
            for adsatom in adsorbates_index[adsorbate_atom]:
                distances[str(el + "-" + adsorbate_atom + str(adsatom))] = round(
                    npmin(struc.get_distances(adsatom, indices[el])), 2
                )

    try:
        bondlength = npmin(list(distances.values()))
    except Exception as error:
        print(row.Name, "An error occured:", type(error).__name__)  # An error occured
        bondlength = 0
    if all_distances:
        return bondlength, all_distances
    else:
        return bondlength


def converged(frame, force_col="fmax", convergence_threshold=0.01):
    """returns a frame with the "fmax" lower than "convergence_threshold" but not 0."""
    return frame[(frame[force_col].lt(convergence_threshold)) & (frame[force_col] != 0)]


def notconverged(frame, force_col="fmax", convergence_threshold=0.01):
    """returns a frame with the converged calculations."""
    return frame[(frame[force_col].gt(convergence_threshold)) | (frame[force_col] == 0)]


def check_constraints(Frame, ref=None , ref_column='surface_ref' , type=None):
    assert (type == "Geometric Optimization") | (type == 'Constrained Optimization'), 'set type either "Constrained Optimization" or "Geometric Optimization" '

    if type == 'Geometric Optimization':
        constraints_len = Frame.apply(lambda x: len(x.constraints), axis=1)

        print('FixAtoms is 2:')
        display(Frame[constraints_len == 2].Name)
        
        print('\nnot same numbers of fixed atoms:')
        display(Frame[Frame.apply(lambda x: len(x.constraints) != len(ref.loc[x.surface_ref].constraints), axis=1)].Name)

        print('No fixed atoms')
        display(Frame[Frame.apply(lambda x: len(x.constraints) == 0, axis=1)].Name)

    if type == 'Constrained Optimization':
        constraints_len = Frame.apply(lambda x: len(x.constraints), axis=1)

        print('FixAtoms is not 2:')
        display(Frame[constraints_len != 2].constraints)
        
        print('\nnot same numbers of fixed atoms:')
        subset = Frame[constraints_len == 2]
        problems =  subset[subset.apply(lambda x: len(x.constraints[1].get_indices()) != len(ref.loc[x.surface_ref].constraints[0].get_indices()), axis=1)]
        display(problems.apply(lambda x: len(x.constraints[1].get_indices()), axis=1) )
        print('No fixed atoms')
        display(Frame[Frame.apply(lambda x: len(x.constraints) == 0, axis=1)].constraints)



def adsorbed(row, Bondlength=2.5):
    """
    Returns True or False depending on the bond distance.
    Needs column with name "Distance". From function "distance_from_surface".
    """
    if Bondlength < row["Distance"]:
        return False
    else:
        return True


def getVacuum(row, structure_column="struc", axis=3):
    """Vacuum between the periodic images in axis 1,2 or 3."""
    positions = row[structure_column].get_positions()
    if len(positions) == 0:
        print("0 atoms in:", row.Name)
        vac = 0
    else:
        vac = (
            row.c
            - np.max(positions, axis=0)[axis - 1]
            + np.min(positions, axis=0)[axis - 1]
        )
    return vac


def frequency(row, sliced=slice(None, None, None), xyzfile="vib.xyz", min=None):
    files = listdir(row.Path)#row["files"]

    frequencies = []
    if xyzfile in files:
        file = open(row.Path + "/" + xyzfile, "r", encoding="utf-8")
        for line in file:
            if re.search("Mode", line):
                # print(line)
                v = line.split(" ")[4]
                if "i" == v[-1]:
                    frequencies.append(float(v[0:-1]) * I)
                else:
                    frequencies.append(float(v))
        frequencies = sorted(
            frequencies,
            reverse=False,
            key=lambda x: (real(x), im(x) if hasattr(x, "is_real") else x),
        )
        if min is None:
            pass
        else:
            frequencies = [val for val in frequencies if im(val) == 0]
            frequencies = np.array(frequencies)
            frequencies = frequencies[frequencies > min]
    # else:
    #    print(xyzfile, 'Not in files ', row.Name)

    return frequencies[sliced]

def get_stochiometric_change(Frame, ref_frame, ref_column='surface_ref'):
    # Merge the original Frame with the reference DataFrame using 'adsorbate_ref' as the key
    merged_df = Frame.join(ref_frame, on=ref_column, rsuffix='_ref')

    # Get all elements
    elements = get_all_elements(Frame)

    # Loop through each element to calculate the delta and add it to Frame
    for el in elements:
        Frame['delta_' + el] = merged_df[el] - merged_df[el + '_ref']
        
    return Frame

def Frequency(Frame, sliced=slice(-1, -5, -1), xyzfile="vib.xyz", min=None):
    Frame["Frequency"] = Frame.apply(frequency, args=[sliced, xyzfile, None], axis=1)


def Imaginary(Frame):
    Frame = Frame.copy()

    def condition(x):
        return [val for val in x if im(val) > 0]

    Frame["Imaginary"] = Frame["Frequency"].apply(condition)
    return Frame[Frame.Imaginary.apply(len) != 0].Imaginary


def lines_that_start_with(string, fp):
    with open(fp, "r", errors="replace") as f:
        try:
            return [line for line in f if line.startswith(string)][-1]
        except Exception as error:
            print("An error occured:", type(error).__name__)  # An error occured
            return NaN


# Entropies


def get_entropies(frame, out_file: str = None, verbose=False):
        
    assert out_file is not None, 'No out_file chosen in get_entropies(frame, out_file=)'

    if verbose:
        logger = logging.getLogger('get_entropies')
        logger.setLevel(logging.DEBUG)
    else:
        logger = logging.getLogger('get_entropies')
        logger.setLevel(logging.INFO)
    tqdm.pandas(desc="reading entropies ") ## creates the progress bar
    entropies = DataFrame(
        columns=[
            "modes_for_G",
            "Cv_at_T",
            "E_pot",
            "E_ZPE",
            "Cv_trans",
            "Cv_rot",
            "Cv_vib",
            "C_vtoC_p",
            "S_trans",
            "S_rot",
            "S_elec",
            "S_vib",
            "Sbar",
            "S",
        ],
        index=frame.index,
    )
    ( entropies["modes_for_G"],
        entropies["Cv_T_at"],
        entropies["E_pot"],
        entropies["E_ZPE"],
        entropies["Cv_trans"],
        entropies["Cv_rot"],
        entropies["Cv_vib"],
        entropies["C_vtoC_p"],
        entropies["S_trans"],
        entropies["S_rot"],
        entropies["S_elec"],
        entropies["S_vib"],
        entropies["Sbar"],
        entropies["S"],
    ) =   zip(*frame.progress_apply(get_zpe_entropies, args=[verbose, out_file], axis=1))
    frame.update(entropies)
    try:
        frame = frame.join(entropies)
    except Exception:
        frame.update(entropies)
    frame['has_entropy'] = frame.S.isna()
    return frame

def find_matching_line_in_file(f: IO, search_string):
    matches = [i for i, line in enumerate(f.read().split("\n")) if search_string == line ]  
    return matches


def Entropylines(fp):
    Entropies = {}
    if not ospath.exists(fp):
        #logging.critical("no file "+ fp)
        return Entropies
    else:
        #logging.debug('file exists: ' +fp)
        with open(fp, "r", errors="replace") as f1:
            matches = find_matching_line_in_file(f1, "  #    meV     cm^-1")
        if len(matches) > 0:
            try:
                with open(fp, "r", errors="replace") as f:
                    for line in f:
                        for m in range(int(matches[-1])):
                            next(f)
                        line = next(f)
                        line = next(f)
                        modes = []

                        while not line.startswith("-"):
                            modes.append(line.split()[2])
                            line = next(f)
                        Entropies["modes_for_G"] = modes

                        while not line.startswith("Enthalpy components at"):
                            line = next(f)
                        if line.startswith("Enthalpy components at T"):
                            Entropies["Cv_T_at"] = float(line.split()[5])
                            line = next(f)

                        while not line.startswith("E_pot"):
                            line = next(f)
                        if line.startswith("E_pot"):
                            Entropies["E_pot"] = float(line.split()[1])
                            line = next(f)

                        while not line.startswith("E_ZPE"):
                            line = next(f)
                        if line.startswith("E_ZPE"):  # Loop until it finds E_pot
                            # then continues the next condition
                            Entropies["E_ZPE"] = float(line.split()[1])
                            line = next(f)

                        while not line.startswith("Cv_trans"):
                            line = next(f)
                        if line.startswith("Cv_trans"):
                            Entropies["Cv_trans"] = float(line.split()[2])
                            line = next(f)

                        while not line.startswith("Cv_rot"):
                            line = next(f)
                        if line.startswith("Cv_rot"):
                            Entropies["Cv_rot"] = float(line.split()[2])
                            line = next(f)

                        while not line.startswith("Cv_vib"):
                            line = next(f)
                        if line.startswith("Cv_vib"):
                            Entropies["Cv_vib"] = float(line.split()[2])
                            line = next(f)

                        while not line.startswith("(C_v -> C_p)"):
                            line = next(f)
                        if line.startswith("(C_v -> C_p)"):
                            Entropies["C_vtoC_p"] = float(line.split()[3])
                            line = next(f)
                        while not line.startswith("S_trans (1 bar)"):
                            #print(line)
                            line = next(f)
                        if line.startswith("S_trans (1 bar)"):
                            Entropies["S_trans"] = float(line.split()[3])
                            line = next(f)

                        if line.startswith("S_rot"):
                            Entropies["S_rot"] = float(line.split()[1])
                            line = next(f)

                        if line.startswith("S_elec"):
                            Entropies["S_elec"] = float(line.split()[1])
                            line = next(f)

                        if line.startswith("S_vib"):
                            Entropies["S_vib"] = float(line.split()[1])
                            line = next(f)

                        if line.startswith("S (1 bar -> P)"):
                            Entropies["Sbar"] = float(line.split()[5])
                            next(f)
                            line = next(f)
                        if line.startswith("S"):
                            Entropies["S"] = float(line.split()[1])
                            next(f)
                            line = next(f)
                        return Entropies
            except StopIteration:
                return Entropies
        else:
            #logger.debug('no matches')
            return Entropies



def get_zpe_entropies(row, verbose=False, out_file=None):
    if verbose:
        logger = logging.getLogger('get_zpe_entropies')

        logger.setLevel(logging.DEBUG)
    else:
        logger = logging.getLogger('get_zpe_entropies')

        logger.setLevel(logging.INFO)

    fp = row["Path"]
    assert out_file is not None, 'No out_file chosen in get_zpe_netropies(out_file)'
    
    entropies = Entropylines(get_pathtofile(fp, out_file))
    #logging.info('output from Entropylines ')
    #logging.info(len(entropies))
    #logging.info(entropies)
    assert entropies is not None, 'None entropies returned'
    if len(entropies) == 0:
        #logger.info(row.Name + " empty entropies")
        return [NaN] * 14
    elif len(entropies) == 14:
        #print(entropies)
        return entropies.values()
    else:
        #logger.info(
        #    row.Name
        #    + " partial entropies: "
        #    + str(len(entropies))
        #    + " "
        #    + str(entropies.keys())
       #)
        return [NaN] * 14

# free G


def gas_free_G(row, T=None):

    T = symbols("T")
    E = row["E"]
    ZPE = row["E_ZPE"]
    Cv_trans = row["Cv_trans"]
    Cv_rot = row["Cv_rot"]
    Cv_vib = row["Cv_vib"]
    S_trans = row["S_trans"]
    S_rot = row["S_rot"]
    S_vib = row["S_vib"]
    G = (
        float(E)
        + float(ZPE)
        + Cv_trans
        + Cv_rot
        + Cv_vib
        - T * (S_trans + S_rot + S_vib)
    )
    if T is None:
        return G
    else:
        return G.subs("T", T)


def ads_free_G(row):
    T = Symbol("T")
    E = row["E"]
    ZPE = row["E_ZPE"]
    Cv_vib = row["Cv_vib"]
    S_vib = row["S_vib"]
    G = float(E) + float(ZPE) + Cv_vib - T * (S_vib)
    return G


def Atommultiindex(Frame, structure_column=None, verbose=False):
    
    assert structure_column is not None, 'The struc_file variable needs the column name that contains the Atoms objects or a filename to read the structure from.'

    if verbose:
        logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger = logging.getLogger('Atommultiindex')
    else:
        logging.basicConfig(level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger = logging.getLogger('Atommultiindex')

    if len(Frame) == 0:
        logger.critical("The Frame is empty")
        return DataFrame(index=["Name", "indices"], columns=["Symbol"])
    if structure_column not in Frame:
        try:
            Frame[structure_column] = Frame.apply(read_strucfile, args=[structure_column], axis=1)
        except Exception as error:
            print("An error occured:", type(error).__name__) 
            logger.critical("could not read" + structure_column)

    def getsymbols(row):
        struc = row[structure_column]
        Name = row.Name
        symbols = struc.get_chemical_symbols()
        names = [Name] * len(struc)
        indices = [n for n in np.arange(0, len(struc))]
        return names, symbols, indices

    nam, sym, ind = zip(*Frame.apply(getsymbols, axis=1))

    names = [i for n in nam for i in n]
    symbols = [i for n in sym for i in n]
    indices = [i for n in ind for i in n]
    # out = pd.MultiIndex(legel=[nam, ind])
    out = DataFrame(symbols, index=[names, indices], columns=["Symbol"])
    out.index.rename = ["Name", "indices"]
    return out


def get_GCN(row, cutoff=3, struc_file="struc"):
    struc = row[struc_file]
    i = neighbor_list("i", struc, cutoff=cutoff)
    unique, counts = np.unique(i, return_counts=True)
    coords = dict(zip(unique, counts))
    j = neighbor_list("j", struc, cutoff=cutoff)
    neighbourcoord = map(lambda x: j[i == x], unique)
    gcn = [sum([coords[d] for d in m]) / 12 for m in neighbourcoord]
    gcn = list(map(round, gcn, [3] * len(counts)))
    return gcn


def GCN(Frame, AtomsFrameIndex):
    gcn_tables = Frame.apply(get_GCN, axis=1)
    gcn_column = [i for m in gcn_tables.to_list() for i in m]
    AtomsFrameIndex["GCN"] = gcn_column
    AtomsFrameIndex["GCN"]


def get_Moments_Frame(Frame, index):
    def getMoments(row):
        struc = row["struc"]
        try:
            Moments = struc.get_magnetic_moments()
        except Exception as error:
            print(
                row.Name, "An error occured:", type(error).__name__
            )  # An error occured
            Moments = [0] * len(struc)
        Name = row[index]
        names = [Name] * len(Moments)
        indices = [n for n in np.arange(0, len(Moments))]
        return names, Moments, indices

    nam, Moments, ind = zip(*Frame.apply(getMoments, axis=1))
    if not len(Moments) == len(ind):
        print(nam)
    names = [i for n in nam for i in n]
    gc = [i for n in Moments for i in n]
    indices = [i for n in ind for i in n]
    # out = pd.MultiIndex(legel=[nam, ind])
    out = pd.DataFrame(gc, index=[names, indices], columns=["Moments"])
    # out.index.rename = ['Name', 'indices']
    return out


######
# Inputparameters
######


def InputParameters(row, filename="struc"):
    if ospath.exists(row["Path"] + "/calc.traj"):
        try:
            a = aseread(row["Path"] + "/calc.traj")
            a = a.calc.parameters
            return a
        except Exception as error:
            print(
                row.Name, "An error occured:", type(error).__name__
            )  # An error occured
            print(row.Name, "calc.traj exception")
            return {}
    else:
        try:
            a = row[filename].calc.parameters
            return a
        except AttributeError as error:
            print(
                row.Name, "Calculator not saved:", type(error).__name__
            )  # An error occured
            print(row.Name, "struc.calc exception")
            return {}
        except Exception as error:
            print(
                row.Name, "An error occured:", type(error).__name__
            )  # An error occured
            print(row.Name, "struc.calc exception")
            return {}


def getparameter(row, parameter, dic_column="parameters"):
    """
    parameter: The calculation parameter to extract. e.g 'kpts', 'encut'
    dic_column: The column that contains the dictionary from reading the INCAR or the Atoms object.
    """
    parameters = row[dic_column]
    try:
        return str(parameters[parameter])
    except Exception as error:
        print(row.Name, "An error occured:", type(error).__name__)  # An error occured
        return 0


def checkforparameter(Frame, parameter, value):
    Frame[parameter] = Frame.apply(getparameter, args=[parameter], axis=1)
    # print(Frame[Frame[parameter] != value]['Path'].to_string(index=False).replace('/Users/dk2994/Desktop/Uni/Calculations', ''))
    # return Frame[Frame[parameter] != value]['Path']
    print(Frame[Frame[parameter] != value][parameter].to_string())


def read_incar(row, filename="INCAR"):
    from pymatgen.io import vasp  # importing here keeps the pymatgen package optional.

    try:
        incar = vasp.inputs.Incar.from_file(get_pathtofile(row.Path, filename))
    except Exception as error:
        print(row.Name, "An error occured:", type(error).__name__)  # An error occured
        return
    return incar

######
#Positions
######
def Positions_2_Multiindex(Frame, structure_column):
    Multiindex = Atommultiindex(Frame, 'struc')
    pos = Frame.apply(lambda x: x[structure_column].get_positions(), axis=1)

    # Find the common indices between Series and Multiindex
    common_indices = pos.index.intersection(Multiindex.unstack().index)
    Multiindex = Multiindex.loc[common_indices]

    # Update Multiindex only for the common indices
    if not common_indices.empty:
        for i in tqdm(common_indices):
            # Directly update using bulk assignment
            try:
                Multiindex.loc[i, ['X', 'Y', 'Z']] = pos.loc[i]
            except Exception as error:
                pass
                print(i, error)
    # Handle the case where the index from pos is not in Multiindex
    missing_indices = pos.index.difference(Multiindex.unstack().index)
    if not missing_indices.empty:
        print(missing_indices, len(missing_indices))
        for i in missing_indices:
            print(f"{i} unknown index Problem")
    return Multiindex


#######
# Bader Charge
#######


def checkxyz(Frame, badertable, verbose, struc='CONTCAR'):
    logger = logging.getLogger('checkxyz')

    def xyzcheck(row, badertable, verbose, struc='CONTCAR'):
        i = row.Name
        j = row
        if badertable.loc[i] is None:
            #print(i, " has no ACF.dat")
            return 'no entry in badertable'
        if len(badertable.loc[i]) == 0:
            return 'ACF.dat not available'
        x1 = badertable.loc[i].X.apply(lambda x: float(x)).to_numpy()
        x2 = j[struc].get_positions()[:, 0]
        y1 = badertable.loc[i].Y.apply(lambda x: float(x)).to_numpy()
        y2 = j[struc].get_positions()[:, 1]
        z1 = badertable.loc[i].Z.apply(lambda x: float(x)).to_numpy()
        z2 = j[struc].get_positions()[:, 2]
        if len(x2) == 0:
            return 'unavailable positions in ' +struc
        if len(x1) != len(x2):
            return 'unequal number of atoms in ACF.dat and struc'
        try:
            delta = sum(
                np.round(x1 - x2, 4) + np.round(y1 - y2, 4) + np.round(z1 - z2, 4)
            )
            
        except Exception as error:
            if verbose:
                print('ACF.dat: ', len(x1), len(y1), len(z1))
                print('struc:', len(x2), len(y2), len(z2))
                logger.debug("An error occured: i : "+i+' '+ type(error).__name__)  # An error occured
            delta = 0
        if np.abs(delta) < 0:
            return delta
        else:
            return delta

    checkreturn = Frame.apply(xyzcheck, args=[badertable, verbose, struc], axis=1)
    return checkreturn


def Charge(Frame, compare=True, verbose=False, struc='CONTCAR'):
    '''
    Collects the charges from ACF.dat into a list of lists.
    If compare is true the x,y,z coordinates from ACF.dat and CONTCAR are compared.
    if verbose true, messages of reading errors, not existing or empty files are reported.
    '''
    badertable = Frame.apply(read_bader, args=[verbose], axis=1)
    # print(badertable)
    if compare:
        check_returns = DataFrame(index=Frame.index)
        check_returns['check_return'] = checkxyz(Frame, badertable, verbose, struc)

        def split_by_type(df, column):
            '''
            Split a dataframe accoding to the type in a column. 
            Either numerics, like integers or floats, or everything else.'''
            is_numeric = df[column].apply(lambda x: isinstance(x, (int, float)))
            numeric_df = df[is_numeric].reset_index(drop=True)
            non_numeric_df = df[~is_numeric].reset_index(drop=True)
            return numeric_df, non_numeric_df
        
        numerics, non_numerics = split_by_type(check_returns, 'check_return')

        display('Unavailable data errors:')
        display(non_numerics.groupby('check_return').value_counts())

        summary = DataFrame(index=['delta:'], data={
        "delta coord. <1": (numerics['check_return'] < -1).sum(),
        "matching coord.": (numerics['check_return'] == 0).sum(),
        "delta coord. >1": (numerics['check_return'] > 1).sum()
        })

        display('Missmatch of positions:', summary)
        display(summary)

        #print(check_returns)
    return badertable


def read_bader(row, verbose=False) -> DataFrame:
    logger = logging.getLogger('read_bader')

    if ospath.exists(get_pathtofile(row.Path, "ACF.dat")):
        try:
            CHG = read_csv(
                row["Path"] + "/ACF.dat",
                skiprows=lambda x: x in [1],
                index_col=None,
                delim_whitespace=True,
            )
        except Exception as error:
            if verbose:
                logger.debug('excepts:', error)
            #print(row.Name, "error reading ACF.dat file:", type(error).__name__)  # An error occured
            #print(row["Path"], "no ACF.dat")
            #return print(row["Path"], row["files"])
            CHG =  pd.DataFrame(columns=['#', 'X', 'Y', 'Z', 'CHARGE', 'MIN', 'DIST', 'ATOMIC', 'VOL'])
    else:
        if verbose:
            logger.debug(row.Name+ ' ACF.dat not found')
        CHG = pd.DataFrame(columns=['#', 'X', 'Y', 'Z', 'CHARGE', 'MIN', 'DIST', 'ATOMIC', 'VOL'])

    CHG.index.name = "index"
    # print(CHG.to_string())
    #    TotalElectron = CHG[CHG["#"] == "NUMBER"].Z.values[0]
    CHG = (CHG.rename(columns={"MIN": "MIN DISTANCES", "DIST": "ATOMIC VOL"})
            .drop(["ATOMIC", "VOL"], axis=1)
            .drop(CHG[-4:].index.to_list())
    )
    try:
        CHG = CHG.astype(float)
    except Exception as error:
        if verbose:
            logger.debug(row.Name+' '+ error)
        pass
    return CHG


# Reaction pathways

def barrier(x1, x2, x3, y1, y2, y3, color="black"):
    #        import numpy as np

    """
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    """

    def fx_parabola(x1, x2, x3, y1, y2, y3):
        """Returns the function of a parabola going through three points:
        y  =  a * x^^2 + b * x + c."""
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
        c = (
            x2 * x3 * (x2 - x3) * y1
            + x3 * x1 * (x3 - x1) * y2
            + x1 * x2 * (x1 - x2) * y3
        ) / denom

        return a, b, c

    def parabolaPoints(x1, x2, a, b, c):
        x_pos = np.arange(x1, x2, 0.001)
        y_pos = []
        # Calculate y values
        for x in range(len(x_pos)):
            x_val = x_pos[x]
            y = (a * (x_val**2)) + (b * x_val) + c
            y_pos.append(y)
        return x_pos, y_pos

    # Plot the parabola (+ the known points)

    # First half of the parabola
    a, b, c = fx_parabola(x1, x2, x3, y1, y2, y1)
    x_pos1, y_pos1 = parabolaPoints(x1, x2, a, b, c)
    plt.plot(x_pos1, y_pos1, linestyle="-", color=color)  # first half of parabola line

    # Second half of the parabola
    a, b, c = fx_parabola(x1, x2, x3, y3, y2, y3)
    x_pos2, y_pos2 = parabolaPoints(x2, x3, a, b, c)
    plt.plot(x_pos2, y_pos2, linestyle="-", color=color)  # second half of parabola line
    #plt.scatter([x1, x2, x3], [y3, y2, y3])
    # plt.scatter(x1,y1,color='r',marker="D",s=50) # 1st known xy
    # plt.scatter(x2,y2,color='g',marker="D",s=50) # 2nd known xy
    # plt.scatter(x3,y3,color='k',marker="D",s=50) # 3rd known xy



# Adsorbate energy

def sum_of_multiple_pairs(df, col_pairs):
    """
    Returns the sum of row-wise multiplications for multiple column pairs in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        col_pairs (list of tuples): A list of tuples where each tuple contains two column names 
                                    to be multiplied and summed.
    
    Returns:
        float: The sum of the row-wise multiplications of each pair.
    """
    total_multiplication = 0
    
    # Loop through each pair of columns
    for col1, col2 in col_pairs:
        # Multiply the values of the pair row-wise
        pair_product = df[col1] * df[col2]
        # Add the sum of the products to the total sum
        total_multiplication += pair_product
    
    return total_multiplication
