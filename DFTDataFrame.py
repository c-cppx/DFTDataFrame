"""Functions for the DataFrame creation."""

import logging
import re
import time
from os import listdir
from os import path as ospath
from os import walk
from os.path import getmtime

import numpy as np
from ase import Atoms
from ase.io import read as aseread
from IPython.display import display
from numpy import NaN
from pandas import DataFrame


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


def crawl(root='/Users/dk2994/Desktop/Uni/Calculations/', flag='out.txt'):
    """Crawl through the subfolders of the given root and returnn the paths that contain the flag file."""
    paths = []
    for path, _folders, files in walk(root):
        if flag in files:
            paths.append(path)
    print(f'Found {len(paths)} folders in {root}')
    return paths


def makename(path, root, dropstrings=None):
    """Creates the index from the path string"""
    name = path.replace(root, '').replace('/', '-')  # root
    if dropstrings is not None:
        for string in dropstrings:
            name = name.replace(string, '')
    name = name.replace('--', '-')
    return name


def read_OUTCAR(row):
    """aseread OUTCAR if it exists and is not empty"""
    if ospath.exists(row['Path'] + '/OUTCAR'):
        try:
            a = aseread(row['Path'] + '/OUTCAR')
            return a
        except IndexError:
            logging.info(row.Path + 'exists but empty')
        except Exception:
            logging.info(row.Path + 'exists but could not be read')
    else:
        return Atoms()


def read_CONTCAR(row):
    """aseread CONTCAR if it exists and is not empty"""
    if ospath.exists(row['Path'] + '/CONTCAR'):
        try:
            return aseread(row['Path'] + '/CONTCAR')
        except IndexError:
            logging.info(row.Path + 'exists but empty')
        except Exception:
            logging.info(row.Path + 'exists but could not be read')
    else:
        return Atoms()


def read_relaxed_structure(row, calc_file='OUTCAR', verbose=False):
    """Read the relaxed structure from OUTCAR, .traj or other ase compatible files.
    Returns the structure as atoms object, cellparameters(a,b,c,gamma) and final energy.

    :param calc_file: output file of geometric relaxation calculation, defaults to OUTCAR
    :type calc_file: str()
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """

    path = row['Path']
    # print([path +'/'+ n for n in os.listdir(path)])
    verboseprint = getverboseprint(verbose)

    timestamp = getmtime(path)
    energy = 0
    calc = 0
    fmax = 0
    cell_a = 0
    cell_b = 0
    cell_c = 0
    gamma = 0
    formula = 0
    pathtofile = ospath.join(path, calc_file)
    if ospath.exists(pathtofile):
        try:
            calc = aseread(pathtofile)
        except Exception:
            verboseprint('{0} could not be read '.format(calc_file))
            calc = Atoms()
        else:
            try:
                energy = calc.get_potential_energy()
                fmax = round(np.max(np.abs(calc.get_forces())), 4)
            except RuntimeError:
                logging.critical('%s energy could not be read', pathtofile)
                energy = 0
                fmax = 0
            else:
                cell_a = calc.cell.cellpar()[0]
                cell_b = calc.cell.cellpar()[1]
                cell_c = calc.cell.cellpar()[2]
                gamma = calc.cell.cellpar()[5]
                formula = calc.get_chemical_formula()
    else:
        logging.info('%s does not exist', pathtofile)
        calc = Atoms()
    if calc == 0:
        logging.error('Outstanding error')
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


# def read_structure_energies(
#     row, calc_file='OUTCAR', struc_file='CONTCAR', verbose=False, dropstrings=None
# ):
#     """aseread the structure files OUTCAR, CONTCAR or others.
#     Expand the Dataframe with the structure, cellparameters and final energy."""
#     path = row['Path']
#     if dropstrings is None:
#         dropstrings = []
#     verboseprint = getverboseprint(verbose)

#     timestamp = getmtime(path)
#     energy = 0
#     struc = 0
#     fmax = 0
#     timestamp = 0
#     cell_a = 0
#     cell_b = 0
#     cell_c = 0
#     gamma = 0
#     formula = 0
#     try:
#         calc = aseread(ospath.join(path, calc_file))
#     except FileNotFoundError:
#         verboseprint(calc_file + ' could not be read')
#         calc = None

#     try:
#         struc = aseread(ospath.join(path, struc_file))
#     except FileNotFoundError:
#         verboseprint(struc_file + ' could not be read')
#         struc = None

#     if struc is None and calc is None:
#         energy = NaN
#         verboseprint(path + ' ' + calc_file + ' could not be read')

#     elif struc is None:
#         print(path + ' ' + struc_file + ' could not be read')
#         try:
#             energy = calc.get_potential_energy()
#             fmax = round(np.max(np.abs(calc.get_forces())), 4)

#         except RuntimeError:
#             verboseprint(
#                 path
#                 + ' '
#                 + struc_file
#                 + ' could not be read and energy could not be read'
#             )
#             energy = 0
#             fmax = 0
#     elif calc is None:
#         logging.info(path + ' ' + struc_file + ' could not be read')
#     else:
#         try:
#             energy = calc.get_potential_energy()
#             fmax = round(np.max(np.abs(calc.get_forces())), 4)
#         except RuntimeError:
#             verboseprint(calc_file + 'Energy could not be obtained')
#             energy = 0
#             fmax = 0

#         path = row.Path
#         timestamp = ospath.getmtime(ospath.join(path))
#         cell_a = struc.cell.cellpar()[0]
#         cell_b = struc.cell.cellpar()[1]
#         cell_c = struc.cell.cellpar()[2]
#         gamma = struc.cell.cellpar()[5]
#         # positions =struc.get_positions()
#         # print(np.max(positions, axis=0)[2] , np.min(positions, axis=0)[2])

#     return (
#         energy,
#         struc,
#         fmax,
#         timestamp,
#         cell_a,
#         cell_b,
#         cell_c,
#         gamma,
#         formula,
#     )


def create_frame(
    root,
    flag_file='out.txt',
    calc_file='OUTCAR',
    droplist=None,
    verbose=False,
):
    """
    - flag: The file to look for when walking the folders that contain calculations.
    - calc: The structure file that contains propiertes from the calculation (calc, final.traj). read with ase.io.aseread
    - struc: The final structure after relaxation. aseread with ase.io.aseread
    """
    if droplist is None:
        droplist = []
    paths = crawl(root=root, flag=flag_file)

    frame = DataFrame(paths, columns=['Path'])
    frame['Name'] = frame['Path'].apply(makename, args=[root, droplist])
    frame = DataFrame(
        frame.Path.to_list(), index=frame['Name'].to_list(), columns=['Path']
    )
    frame['Name'] = frame.index
    frame['files'] = frame['Path'].apply(listdir)
    if verbose:
        display(frame)
    (
        frame['E'],
        frame['struc'],
        frame['fmax'],
        frame['timestamp'],
        frame['a'],
        frame['b'],
        frame['c'],
        frame['gamma'],
        frame['Formula'],
    ) = zip(
        *frame.apply(
            read_relaxed_structure,
            args=[calc_file, verbose],
            axis=1,
        )
    )

    return DataFrame(frame)


def update(
    frame,
    root,
    calc_file='OUTCAR',
    flag_file='out.txt',
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
            return time.ctime(row.timestamp) < time.ctime(getmtime(row.Path))
        except FileNotFoundError:
            logging.info('Path does not exist ' + row.Path)
            return False

    def nonexistent_paths(row):
        """remove from Frame if the path does not exist anymore."""
        return ospath.exists(row.Path)

    def remove_nonexistent_paths(frame):
        nonexistant = frame[~frame.apply(nonexistent_paths, axis=1)].index
        logging.info('%s removed', len(nonexistant))
        return frame.drop(nonexistant)

    newpaths = crawl(root, flag_file)

    for i, row in frame.iterrows():
        if row.Path in newpaths:
            newpaths.remove(row.Path)
        else:
            logging.info(i, ' was moved or renamed')
    modifiedpaths = frame[frame.apply(is_modified, axis=1)].Path.to_list()
    paths_to_update = modifiedpaths + newpaths

    if paths_to_update:
        logging.info('update %s', len(paths_to_update))

        new = DataFrame(paths_to_update, columns=['Path'])
        new['Name'] = new['Path'].apply(makename, args=[root, droplist])

        new = DataFrame(new.Path.to_list(),
                        index=new['Name'].to_list(), columns=['Path'])
        print(new)
        new['files'] = new['Path'].apply(listdir)

        new['E'],
        new['struc'],
        new['fmax'],
        new['timestamp'],
        new['a'],
        new['b'],
        new['c'],
        new['gamma'],
        new['Formula'] = zip(
            *new.apply(read_relaxed_structure, args=[calc_file, verbose, droplist], axis=1)
        )
        new['files'] = new['Path'].apply(listdir)

        for i in new.index.to_list():  # remove lines from frame which are added with the new data
            if i in frame.index():
                frame.drop(i, inplace=True)

        frame = frame.append(new)
    else:
        logging.info('Nothing to update')

    frame = remove_nonexistent_paths(frame)

    return frame


def surface(Frame):
    index = Frame.index
    Frame.loc[index, 'B_clean'] = [
        True if 'clean' in i else False for i in index]
    Frame.loc[index, 'B_convergence'] = [
        True if 'convergence' in i else False for i in index
    ]
    Frame.loc[index, 'B_adsorbate'] = [
        True if 'NH3' in i else False for i in index]

    for i in [
        'CHO',
        'NH3',
        'NH2_H' 'CuO',
        'NiO',
        'H',
        'Hx4',
        'O',
        'Ox2',
        'Ox3',
        'OH',
        'OHx2',
        'HCOO',
        'COOH',
        'CO',
        'CO2',
        'COx2',
        'Hx2',
        'CN',
        'CNx2',
    ]:
        if '-' + i + '-' in index:
            Frame.loc[index, 'B_adsorbate'] = True
            continue


def Adsorption(
    Frame,
    Adsorbates=None,
):
    # Area = a*b*np.sin(np.radians(gamma))
    # vac = c-np.max(positions, axis=0)[2] + np.min(positions, axis=0)[2]#
    if Adsorbates is None:
        Adsorbates = [
            'CHO',
            'NH3',
            'NH2_H' 'CuO',
            'NiO',
            'H',
            'Hx4',
            'O',
            'Ox2',
            'Ox3',
            'OH',
            'OHx2',
            'HCOO',
            'COOH',
            'CO',
            'CO2',
            'COx2',
            'Hx2',
            'CN',
            'CNx2',
        ]
    index = Frame.index
    Frame.loc[index, 'B_clean'] = [
        True if 'clean' in i else False for i in index]
    Frame.loc[index, 'B_adsorbate'] = [
        True if 'NH3' in i else False for i in index]

    for i in [
        'CHO',
        'NH3',
        'NH2_H' 'CuO',
        'NiO',
        'H',
        'Hx4',
        'O',
        'Ox2',
        'Ox3',
        'OH',
        'OHx2',
        'HCOO',
        'COOH',
        'CO',
        'CO2',
        'COx2',
        'Hx2',
        'CN',
        'CNx2',
    ]:
        if '-' + i + '-' in index:
            Frame.loc[index, 'B_adsorbate'] = True
            continue


#    def percent_frameoy(row):
#       per = row[row['MO']] / (row[row['M']]+row[row['MO']])
#       return round(per, 4)*100

#    def getSurfaceframeoy(Name):
#        return '-'.join(Name.split('-')[0:3])

#   def getSlabInfo(row):
#       Name = row['Name']
#       M = row['M']
#       Slabsize = '-'.join(Name.split('-')[3:4])
#       hkl = '-'.join(Name.split('-')[1:2])
#       #print(M, hkl, Slabsize)
#       Slabname = '-'.join([M, hkl, Slabsize])
#       return Slabname, Slabsize, hkl


# B_adsorbate = True if 'C' in row['final_traj'].symbols.indices().keys() else False

# ML = apply(lambda row: row['Ni'] + row['Zn'], axis=1)
# ML = apply(getML, axis=1)


def Surfaceframeoy(Frame):
    def getML(row):
        hkl = row['hkl']
        Slabsize = row['Slab_size']
        layers = Slabsize.split('x')[-1]
        # print(row)
        if row['Ni'] == 0:
            nM = row['Cu']
        else:
            nM = row['Ni']
        if row['Ga'] == 0:
            nMO = row['Zn']
        else:
            nMO = row['Ga']
        if nMO * nM == 0:
            # print(row['Name'])
            return 0
        else:
            # print('ML', nMO, nM)
            if '211' in hkl:
                return round(nMO / ((nM + nMO) / 12), 4)
            else:
                try:
                    layers = int(layers)
                except Exception:
                    layers = 4
                return round(nMO / ((nM + nMO) / int(layers)), 4)


def converged(Frame):
    return Frame[(Frame['fmax'].lt(0.01)) & (Frame['fmax'] != 0)]


def notconverged(Frame):
    return Frame[(Frame['fmax'].gt(0.01)) | (Frame['fmax'] == 0)]


def adsorbed(row):
    if 2.5 < row['Distance']:
        return False
    else:
        return True


def frequency(row):
    #    '/Users/dk2994/Desktop/Uni/scripts
    #   atoms = row['final_traj']
    path = row.root + row['Path']
    files = row['files']
    # print(path)
    # os.chdir(row['Path'])
    Freq = []
    if 'vib.xyz' in files:
        file = open(path + '/vib.xyz', 'r')
        for line in file:
            if re.search('Mode', line):
                # print(line)
                Freq.append(line.split(' ')[4])
    return Freq[-6:]


def Frequency(Frame):
    Frame['Frequency'] = Frame.apply(frequency, axis=1)


def lines_that_start_with(string, fp):
    with open(fp, 'r') as f:
        try:
            return [line for line in f if line.startswith(string)][-1]
        except Exception:
            return NaN


def Zeropointenergy(fp):
    zpe = lines_that_start_with('Zero-point energy:', fp)
    try:
        return zpe.split(' ')[2]
    except Exception:
        return NaN


def get_zpe(row):
    fp = row['Path']
    try:
        zpe = Zeropointenergy(fp + '/vib.out')
    except Exception:
        zpe = Zeropointenergy(fp + '/out.txt')
    return zpe


def get_entropies(frame):
    entropies = DataFrame(
        columns=[
            'E_pot',
            'E_ZPE',
            'Cv_trans',
            'Cv_rot',
            'Cv_vib',
            'C_vtoC_p',
            'S_trans',
            'S_rot',
            'S_elec',
            'S_vib',
            'Sbar',
            'S',
        ],
        index=frame.index,
    )
    (
        entropies['E_pot'],
        entropies['E_ZPE'],
        entropies['Cv_trans'],
        entropies['Cv_rot'],
        entropies['Cv_vib'],
        entropies['C_vtoC_p'],
        entropies['S_trans'],
        entropies['S_rot'],
        entropies['S_elec'],
        entropies['S_vib'],
        entropies['Sbar'],
        entropies['S'],
    ) = zip(*frame.apply(get_zpe_entropies, axis=1))
    try:
        frame = frame.join(entropies)
    except Exception:
        frame.update(entropies)

    return frame


def Entropylines(fp):
    Entropies = {}
    with open(fp, 'r') as f:
        try:
            for line in f:
                if line.startswith(
                    'E_pot'
                ):  # Loop until it finds E_pot then continues the next condition
                    Entropies['E_pot'] = float(line.split()[1])
                    line = next(f)
                else:
                    continue

                if line.startswith(
                    'E_ZPE'
                ):  # Loop until it finds E_pot then continues the next condition
                    Entropies['E_ZPE'] = float(line.split()[1])
                    line = next(f)

                if line.startswith('Cv_trans'):
                    Entropies['Cv_trans'] = float(line.split()[2])
                    line = next(f)

                if line.startswith('Cv_rot (0->T)'):
                    Entropies['Cv_rot'] = float(line.split()[2])
                    line = next(f)

                if line.startswith('Cv_vib (0->T)'):
                    Entropies['Cv_vib'] = float(line.split()[2])
                    line = next(f)

                if line.startswith('(C_v -> C_p)'):
                    Entropies['C_vtoC_p'] = float(line.split()[3])
                    line = next(f)
                while not line.startswith('S_trans (1 bar)'):
                    line = next(f)
                if line.startswith('S_trans (1 bar)'):
                    Entropies['S_trans'] = float(line.split()[3])
                    line = next(f)

                if line.startswith('S_rot'):
                    Entropies['S_rot'] = float(line.split()[1])
                    line = next(f)

                if line.startswith('S_elec'):
                    Entropies['S_elec'] = float(line.split()[1])
                    line = next(f)

                if line.startswith('S_vib'):
                    Entropies['S_vib'] = float(line.split()[1])
                    line = next(f)

                if line.startswith('S (1 bar -> P) '):
                    Entropies['Sbar'] = float(line.split()[5])
                    next(f)
                    line = next(f)
                if line.startswith('S'):
                    Entropies['S'] = float(line.split()[1])
                    next(f)
                    line = next(f)

            return Entropies
        except Exception:
            return Entropies


# Entropy


def get_zpe_entropies(row):
    fp = row['Path']
    try:
        #        zpe = Zeropointenergy(fp + '/vib.out')
        entropies = Entropylines(fp + '/vib.out')
    except Exception:
        #        zpe = Zeropointenergy(fp + '/out.txt')
        entropies = Entropylines(fp + '/out.txt')
    if len(entropies) == 12:
        return tuple(entropies.values())
    else:
        print(fp, '\n', entropies)
        return [NaN] * 12


def gas_free_G(row, T=None):
    from sympy import symbols

    kb = 8.617333262145e-5
    Temp = symbols('Temp')
    E = row['E']
    ZPE = row['E_ZPE']
    Cv_trans = row['Cv_trans']
    Cv_rot = row['Cv_rot']
    Cv_vib = row['Cv_vib']
    S_trans = row['S_trans']
    S_rot = row['S_rot']
    S_vib = row['S_vib']
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
        return G.subs('Temp', T)


def Atommultiindex(Frame):
    def getsymbols(row):
        struc = row['struc']
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
    out = DataFrame(symbols, index=[names, indices], columns=['Symbols'])
    out.index.rename = ['Name', 'indices']
    return out
