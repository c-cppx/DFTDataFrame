"""Functions for the DataFrame creation."""

import logging
import re
from os import listdir
from os import path as ospath
from os import walk
from os.path import getmtime

import numpy as np
from sympy import Symbol, im, I
from sympy import re as real
from sympy import symbols

from ase import Atoms
from ase.io import read as aseread
from IPython.display import display
from numpy import NaN
from numpy import min as npmin
from pandas import DataFrame, read_csv
import pandas as pd
from pathlib import Path


from ase.neighborlist import neighbor_list


logging.basicConfig()


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
            print(
                row.Name, "An error occured:", type(error).__name__
            )  # An error occured
            logging.critical(file + " not readable " + row.Name)
            return Atoms()
    else:
        return row.struc


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


def crawl(root="/Users/dk2994/Desktop/Uni/Calculations/", flag="out.txt"):
    """Crawl through the subfolders of the given root and returnn the paths
    that contain the flag file."""
    paths = []
    for path, _folders, files in walk(root, followlinks=True):
        if flag in files:
            paths.append(path)
    if len(paths) == 0:
        print("No " + flag + " found in ", root)
    else:
        print(f"Found {len(paths)} folders in {root}")
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


def fill_struc_gap(row, fillwith):
    """Fills the gaps in a structure column with the structure from a given file."""
    if row.struc == Atoms():
        struc = read_strucfile(row, fillwith)
        try:
            E = struc.get_potential_energy()
            fmax = np.max(struc.get_forces())
        except Exception as error:
            print(
                row.Name, "An error occured:", type(error).__name__
            )  # An error occured
            E = NaN
            fmax = NaN
            print(row.Name, fillwith, "no calculator")
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
    :return: energy, calc, fmax, timestamp, cell_a, cell_b, cell_c, gamma,
      formula,
    :rtype: float, Atoms, float, time.ctime, float, float, float, float, str
    """

    path = row["Path"]
    # print([path +'/'+ n for n in os.listdir(path)])
    verboseprint = getverboseprint(verbose)
    timestamp = getmtime(path)
    energy = 0
    calc = Atoms()
    fmax = 0
    cell_a = 0
    cell_b = 0
    cell_c = 0
    gamma = 0
    formula = 0
    pathtofile = get_pathtofile(path, calc_file)
    if ospath.exists(pathtofile):
        try:
            calc = aseread(pathtofile)
        except Exception as error:
            print(
                row.Name, "An error occured:", type(error).__name__
            )  # An error occured
            verboseprint("{0} could not be read ".format(calc_file))
            calc = Atoms()
        else:
            try:
                energy = calc.get_potential_energy()
                fmax = round(np.max(np.abs(calc.get_forces())), 4)
            except RuntimeError:
                logging.critical("%s energy could not be read", pathtofile)
                energy = 0
                fmax = 0
            else:
                cell_a = calc.cell.cellpar()[0]
                cell_b = calc.cell.cellpar()[1]
                cell_c = calc.cell.cellpar()[2]
                gamma = calc.cell.cellpar()[5]
                formula = calc.get_chemical_formula()
    else:
        logging.info("%s does not exist", pathtofile)
        calc = Atoms()
    if calc == 0:
        logging.error("Outstanding error")
    return (
        energy,
        calc,
        fmax,
        timestamp,
        cell_a,
        cell_b,
        cell_c,
        gamma,
        formula,
    )


def create_frame(
    root,
    flag_file="out.txt",
    calc_file="OUTCAR",
    droplist=None,
    replacedic=None,
    verbose=False,
):
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

    paths = crawl(root=root, flag=flag_file)

    frame = DataFrame(paths, columns=["Path"])
    frame["Name"] = frame["Path"].apply(makename, args=[root, droplist, replacedic])
    frame = DataFrame(
        frame.Path.to_list(),
        index=frame["Name"].to_list(),
        columns=["Path"],
    )
    frame["Name"] = frame.index
    frame["files"] = frame["Path"].apply(listdir)
    if verbose:
        display(frame)
    if len(frame) == 0:
        print(frame)
    else:
        (
            frame["E"],
            frame["struc"],
            frame["fmax"],
            frame["timestamp"],
            frame["a"],
            frame["b"],
            frame["c"],
            frame["gamma"],
            frame["Formula"],
        ) = zip(*frame.apply(read_relaxed_structure, args=[calc_file, verbose], axis=1))

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
            return row.timestamp < getmtime(row.Path)
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

    paths_to_update = modifiedpaths + newpaths
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
    ML_min = DataFrame(columns=Frame.columns)
    for name, group in MLgroup:
        group = MLgroup.get_group(name)
        ML_min = pd.concat([ML_min, group.sort_values(value).head(n=1)])
    return ML_min


def Adsorption(
    frame,
    adsorbates=None,
):
    # Area = a*b*np.sin(np.radians(gamma))
    # vac = c-np.max(positions, axis=0)[2] + np.min(positions, axis=0)[2]#
    if adsorbates is None:
        adsorbates = [
            "CHO",
            "NH3",
            "NH2_H" "CuO",
            "NiO",
            "H",
            "Hx4",
            "O",
            "Ox2",
            "Ox3",
            "OH",
            "OHx2",
            "HCOO",
            "COOH",
            "CO",
            "CO2",
            "COx2",
            "Hx2",
            "CN",
            "CNx2",
        ]
    index = frame.index
    frame.loc[index, "B_clean"] = [True if "clean" in i else False for i in index]
    frame.loc[index, "B_adsorbate"] = [True if "NH3" in i else False for i in index]

    for i in [
        "CHO",
        "NH3",
        "NH2_H",
        "CuO",
        "NiO",
        "H",
        "Hx4",
        "O",
        "Ox2",
        "Ox3",
        "OH",
        "OHx2",
        "HCOO",
        "COOH",
        "CO",
        "CO2",
        "COx2",
        "Hx2",
        "CN",
        "CNx2",
    ]:
        if "-" + i + "-" in index:
            frame.loc[index, "B_adsorbate"] = True
            continue


def distance_from_surface(row, struc=None, adsorbate_atoms=['C', 'O', 'H'] , all_distances=False):
    """Return the distance of the adsorbate from the distance. Atoms with C, O, H are considered part of the adsorbate.
    Returns all distances or the minimum distance of C, O or H to any other element that is not C, O or H."""
    distances = {}
    adsorbates_index = {}
    struc=row[struc]
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
    """returns a frame with the converged calculations."""
    return frame[(frame[force_col].lt(convergence_threshold)) & (frame[force_col] != 0)]


def notconverged(frame, force_col="fmax", convergence_threshold=0.01):
    """returns a frame with the converged calculations."""
    return frame[(frame[force_col].gt(convergence_threshold)) | (frame[force_col] == 0)]


def adsorbed(row, Bondlength=2.5):
    '''
    Returns True or False depending on the bond distance.
    Needs column with name "Distance". From function "distance_from_surface".
    '''
    if Bondlength < row["Distance"]:
        return False
    else:
        return True


def getVacuum(row, axis=3):
    """Vacuum between the periodic images in axis 1,2 or 3."""
    positions = row.struc.get_positions()
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
    path = row["Path"]
    files = row["files"]
    # print(path)
    # os.chdir(row['Path'])
    frequencies = []
    if xyzfile in files:
        file = open(path + "/" + xyzfile, "r", encoding="utf-8")
        for line in file:
            if re.search("Mode", line):
                # print(line)
                v = line.split(" ")[4]
                if "i" == v[-1]:
                    frequencies.append(float(v[0:-1]) * I)
                else:
                    frequencies.append(float(v))
        frequencies = sorted(
            frequencies, key=lambda x: (real(x), im(x) if hasattr(x, "is_real") else x)
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


def Zeropointenergy(fp):
    zpe = lines_that_start_with("Zero-point energy:", fp)
    try:
        return zpe.split(" ")[2]
    except Exception as error:
        print("An error occured:", type(error).__name__)  # An error occured
        return NaN


def get_zpe(row, out_file):
    fp = row["Path"]
    zpe = Zeropointenergy(fp + "/" + out_file)
    return zpe


def get_entropies(frame, out_file="out.txt"):
    entropies = DataFrame(
        columns=[
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
    (
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
    ) = zip(*frame.apply(get_zpe_entropies, args=[out_file], axis=1))
    try:
        frame = frame.join(entropies)
    except Exception as error:
        #print("Overwrite Entropies in Frame")  # An error occured
        frame.update(entropies)

    return frame


def Entropylines(fp):
    Entropies = {}
    if not ospath.exists(fp):
        return "no file"
    else:
        with open(fp, "r") as f:
            try:
                for line in f:
                    if line.startswith("E_pot"):  # Loop until it finds E_pot
                        # then continues the next condition
                        Entropies["E_pot"] = float(line.split()[1])
                        line = next(f)
                    else:
                        continue

                    if line.startswith(
                        "E_ZPE",
                    ):  # Loop until it finds E_pot
                        # then continues the next condition
                        Entropies["E_ZPE"] = float(line.split()[1])
                        line = next(f)

                    if line.startswith("Cv_trans"):
                        Entropies["Cv_trans"] = float(line.split()[2])
                        line = next(f)

                    if line.startswith("Cv_rot (0->T)"):
                        Entropies["Cv_rot"] = float(line.split()[2])
                        line = next(f)

                    if line.startswith("Cv_vib (0->T)"):
                        Entropies["Cv_vib"] = float(line.split()[2])
                        line = next(f)

                    if line.startswith("(C_v -> C_p)"):
                        Entropies["C_vtoC_p"] = float(line.split()[3])
                        line = next(f)
                    while not line.startswith("S_trans (1 bar)"):
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

                    if line.startswith("S (1 bar -> P) "):
                        Entropies["Sbar"] = float(line.split()[5])
                        next(f)
                        line = next(f)
                    if line.startswith("S"):
                        Entropies["S"] = float(line.split()[1])
                        next(f)
                        line = next(f)

                return Entropies
            except Exception as error:
                print(
                    fp, "An error occurred:", type(error).__name__
                )  # An error occurred:
                return Entropies


def get_zpe_entropies(row, out_file="out.txt"):
    fp = row["Path"]
    try:
        #        zpe = Zeropointenergy(fp + '/vib.out')
        entropies = Entropylines(get_pathtofile(fp, out_file))
    except Exception as error:
        print(row.Name, "An error occured:", type(error).__name__)  # An error occured
        #        zpe = Zeropointenergy(fp + '/out.txt')
        entropies = Entropylines(get_pathtofile(fp, out_file))
    if len(entropies) == 12:
        return tuple(entropies.values())
    else:
        print(fp, "\n", entropies)
        return [NaN] * 12


# free G


def gas_free_G(row, T=None):

    kb = 8.617333262145e-5
    Temp = symbols("Temp")
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
        - kb * Temp * (S_trans + S_rot + S_vib)
    )
    if T is None:
        return G
    else:
        return G.subs("Temp", T)


def ads_free_G(row):
    T = Symbol("T")
    E = row["E"]
    ZPE = row["E_ZPE"]
    Cv_vib = row["Cv_vib"]
    S_vib = row["S_vib"]
    G = float(E) + float(ZPE) + Cv_vib - T * (S_vib)
    return G.subs("Temp", 550.15)


def Atommultiindex(Frame, struc_file="CONTCAR"):
    if len(Frame) == 0:
        logging.critical("The Frame is empty")
        return DataFrame(index=["Name", "indices"], columns=["Symbols"])
    if struc_file not in Frame:
        try:
            Frame[struc_file] = Frame.apply(read_strucfile, args=[struc_file], axis=1)
        except Exception as error:
            print("An error occured:", type(error).__name__)  # An error occured
            logging.critical("could not read" + struc_file)

    def getsymbols(row):
        struc = row[struc_file]
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
    out = DataFrame(symbols, index=[names, indices], columns=["Symbols"])
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


def InputParameters(row, filename='struc'):
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
        except Exception as error:
            print(
                row.Name, "An error occured:", type(error).__name__
            )  # An error occured
            print(row.Name, "struc.calc exception")
            return {}


def getparameter(row, parameter, dic_column='parameters'):
    '''
    parameter: The calculation parameter to extract. e.g 'kpts', 'encut'
    dic_column: The column that contains the dictionary from reading the INCAR or the Atoms object.
    '''
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


#######
# Bader Charge
#######


def checkxyz(Frame, badertable):
    for i, j in Frame.iterrows():
        if badertable.loc[i] is None:
            print(i, " has no ACF.dat")
            continue
        if len(badertable.loc[i]) == 0:
            continue
        x1 = badertable.loc[i].X.apply(lambda x: float(x)).to_numpy()
        x2 = j.struc.get_positions()[:, 0]
        y1 = badertable.loc[i].Y.apply(lambda x: float(x)).to_numpy()
        y2 = j.struc.get_positions()[:, 1]
        z1 = badertable.loc[i].Z.apply(lambda x: float(x)).to_numpy()
        z2 = j.struc.get_positions()[:, 2]
        try:
            delta = sum(
                np.round(x1 - x2, 4) + np.round(y1 - y2, 4) + np.round(z1 - z2, 4)
            )
        except Exception as error:
            print(i, "An error occured:", type(error).__name__)  # An error occured
            delta = 0
        if delta == 0:
            print("delta0", i)
            continue
        else:
            print(
                j.Path.replace("/Users/dk2994/Desktop/Uni/Calculations/", "")
            )  # ,#  j['fmax'])


def Charge(Frame):
    badertable = Frame.apply(read_bader, axis=1)
    # print(badertable)
    checkxyz(Frame, badertable)
    return badertable


def read_bader(row):
    try:
        CHG = read_csv(
            row["Path"] + "/ACF.dat",
            skiprows=lambda x: x in [1],
            index_col=None,
            delim_whitespace=True,
        )
    except Exception as error:
        print(row.Name, "An error occured:", type(error).__name__)  # An error occured
        print(row["Path"], "no ACF.dat")
        return print(row["Path"], row["files"])
    CHG.index.name = "index"
    # print(CHG.to_string())
    #    TotalElectron = CHG[CHG["#"] == "NUMBER"].Z.values[0]
    CHG = CHG.rename(columns={"MIN": "MIN DISTANCES", "DIST": "ATOMIC VOL"})
    CHG.drop(["ATOMIC", "VOL"], axis=1, inplace=True)
    CHG.drop(CHG[-4:].index.to_list(), inplace=True)
    return CHG
