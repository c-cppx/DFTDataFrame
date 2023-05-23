

import os
from ase.io import read
import pandas as pd
import numpy as np
import time
from numpy import NaN
import re
from pandas import DataFrame
from pandas import DataFrame as df




def create_Frame(root, flag='out.txt'):
    Frame = []
    for Path, dirs, files in os.walk(root):
        if flag in files:
            try:
                traj = read(os.path.join(Path, 'relax.traj'))
            except:
                traj = None
            try:
                outcar = read(os.path.join(Path, 'OUTCAR'))
            except:
                outcar = None
            if traj is None and outcar is None:
                continue
            try:
                energy = outcar.get_potential_energy()
                struc = outcar
            except:
                try:
                    energy = traj.get_potential_energy()
                    struc= traj
                except:
                    energy = None
            if root[-1] != '/':  #Root should end with / for consistency
                root = root + '/'    
            path = Path.replace(root, '')
            name = path.replace('/', '-')
            fmax =round(np.max(np.abs(struc.get_forces())), 4)
            timestamp = os.path.getmtime(os.path.join(root+path, flag))

            a = struc.cell.cellpar()[0]
            b = struc.cell.cellpar()[1]
            c = struc.cell.cellpar()[2]
            gamma = struc.cell.cellpar()[5]

            #positions =struc.get_positions()
            #print(np.max(positions, axis=0)[2] , np.min(positions, axis=0)[2])
            Formula = (''.join(set(struc.get_chemical_symbols()))
                    .replace('ZnCu', 'CuZn')
                    .replace('ZnNi', 'NiZn')
                    .replace('GaNi', 'NiGa')
            )

            Frame.append({'index': name, 'Name': name, 'Path': path, 'files': os.listdir(os.path.join(root, path)),
                            'E': energy, 'struc': struc, 'final_traj': traj, 'OUTCAR': outcar, 'fmax':fmax, 'timestamp': timestamp, 'root':root, 'a' : a,  'b' : b,  'c' : c,  'gamma' : gamma, 'Formula' : Formula})            
                

    return DataFrame(Frame).set_index('index')

def update(Frame, flag='out.txt'):
    '''
    Flag is the file to look out.
    '''

    Frame['new_timestamp'] = Frame.apply(lambda row: os.path.getmtime(row.root + row.Path + '/'+flag), axis=1)
    mask = Frame['new_timestamp'] > Frame['timestamp']
    for index, row in Frame[mask].iterrows():
        try:
            traj = read(row.root + row.path + 'final.traj')
        except:
            try:
                traj = read(row.root + row.path + 'relax.traj')
            except:
                traj = None
        try:
            outcar = read(row.root + row.path + 'OUTCAR')
        except:
            outcar = None
        if traj is None and outcar is None:
            continue
        try:
            energy = outcar.get_potential_energy()
            struc = outcar
            print('struc is OUTCAR')
        except:
            try:
                energy = traj.get_potential_energy()
                struc = traj
            except:
                energy = None

        Frame.loc[index, 'a'] = struc.cell.cellpar()[0]
        Frame.loc[index, 'b'] = struc.cell.cellpar()[1]
        Frame.loc[index, 'c'] = struc.cell.cellpar()[2]
        Frame.loc[index, 'gamma'] = struc.cell.cellpar()[5]

        Frame.loc[index, 'positions'] =struc.get_positions()
        #print(np.max(positions, axis=0)[2] , np.min(positions, axis=0)[2])
        Frame.loc[index, 'Formula'] = (''.join(set(struc.get_chemical_symbols()))
                .replace('ZnCu', 'CuZn')
                .replace('ZnNi', 'NiZn')
                .replace('GaNi', 'NiGa')
        )
        path = row.path
        Frame.loc[path, 'E'] = energy
        Frame.loc[path, 'final_traj'] = traj
        Frame.at[path, 'Files'] = os.listdir(os.path.join(row.root, path))
        Frame.loc[path, 'OUTCAR'] = outcar
        Frame.loc[path, 'timestamp'] = Frame.loc[path, 'new_timestamp']

        Frame.drop('new_timestamp', axis=1, inplace=True)


 #   def InputParameters(row):
 #       try:
 #           return row['final_traj'].calc.parameters
 #       except:           
 #           return {}




#class Gas(GeometryOptimizationFrameFrame):
#    def __init__( row.root):        
#        super().__init__(row.root)


def Surface(Frame):
    index = Frame.index
    Frame.loc[index, 'B_clean']  = [True if 'clean' in i else False for i in index]
    Frame.loc[index, 'B_convergence']  = [True if 'convergence' in i else False for i in index]
    Frame.loc[index, 'B_adsorbate']  = [True if 'NH3' in i else False for i in index]

    for i in ['CHO', 'NH3', 'NH2_H' 'CuO', 'NiO', 'H', 'Hx4', 'O', 'Ox2', 'Ox3', 'OH', 'OHx2', 'HCOO', 'COOH', 'CO', 'CO2', 'COx2', 'Hx2', 'CN', 'CNx2']:
        if '-'+i+'-' in index:
            Frame.loc[index, 'B_adsorbate']  = True
            continue


def Adsorption(Frame, Adsorbates=['CHO', 'NH3', 'NH2_H' 'CuO', 'NiO', 'H', 'Hx4', 'O', 'Ox2', 'Ox3', 'OH', 'OHx2', 'HCOO', 'COOH', 'CO', 'CO2', 'COx2', 'Hx2', 'CN', 'CNx2']):
     
    #Area = a*b*np.sin(np.radians(gamma))
    #vac = c-np.max(positions, axis=0)[2] + np.min(positions, axis=0)[2]#
    index = Frame.index
    Frame.loc[index, 'B_clean']  = [True if 'clean' in i else False for i in index]
    Frame.loc[index, 'B_adsorbate']  = [True if 'NH3' in i else False for i in index]

    for i in ['CHO', 'NH3', 'NH2_H' 'CuO', 'NiO', 'H', 'Hx4', 'O', 'Ox2', 'Ox3', 'OH', 'OHx2', 'HCOO', 'COOH', 'CO', 'CO2', 'COx2', 'Hx2', 'CN', 'CNx2']:
        if '-'+i+'-' in index:
            Frame.loc[index, 'B_adsorbate']  = True
            continue
#    def percent_alloy(row):
#       per = row[row['MO']] / (row[row['M']]+row[row['MO']])
#       return round(per, 4)*100

#    def getSurfaceAlloy(Name):
#        return '-'.join(Name.split('-')[0:3])

#   def getSlabInfo(row):
#       Name = row['Name']
#       M = row['M']
#       Slabsize = '-'.join(Name.split('-')[3:4])
#       hkl = '-'.join(Name.split('-')[1:2])
#       #print(M, hkl, Slabsize)
#       Slabname = '-'.join([M, hkl, Slabsize])
#       return Slabname, Slabsize, hkl


        

        #B_adsorbate = True if 'C' in row['final_traj'].symbols.indices().keys() else False

    #ML = apply(lambda row: row["Ni"] + row["Zn"], axis=1)   
    #ML = apply(getML, axis=1)


def SurfaceAlloy(Frame):

    def getML(row):
        hkl = row['hkl']
        Slabsize = row['Slab_size']
        layers = Slabsize.split('x')[-1]
        #print(row)
        if row['Ni'] == 0:
            nM = row['Cu']
        else:
            nM = row['Ni']
        if row['Ga'] == 0:
            nMO = row['Zn']
        else:
            nMO = row['Ga']
        if nMO*nM == 0:
            #print(row['Name'])
            return 0
        else:
            #print('ML', nMO, nM)
            if '211' in hkl:
                return round(nMO / ((nM+nMO)/12), 4)
            else:
                try:
                    layers = int(layers)
                except:
                    layers = 4
                return round(nMO / ((nM+nMO)/int(layers)), 4)
            


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
    here='/Users/dk2994/Desktop/Uni/scripts'
    atoms = row['final_traj']
    path = row.root+row['Path']
    files = row['files']
    #print(path)
    #os.chdir(row['Path'])
    Freq=[]
    if 'vib.xyz' in files:
        file = open(path+"/vib.xyz", "r")
        for line in file:
            if re.search('Mode', line):
                #print(line)
                Freq.append(line.split(' ')[4])
    return Freq[-6:]
def Frequency(Frame):
    Frame['Frequency']= Frame.apply(frequency, axis=1)



def lines_that_start_with(string, fp):
    with open(fp, "r") as f:
        try:
            return [line for line in f if line.startswith(string)][-1]
        except:
            return NaN

def Zeropointenergy(fp):
    zpe = lines_that_start_with('Zero-point energy:', fp)
    try:
        return zpe.split(' ')[2]
    except:
        return NaN
    
def get_zpe(row):
    fp = row['Path']
    try:
        zpe =Zeropointenergy(fp+'/vib.out')
    except:
        zpe =Zeropointenergy(fp+'/out.txt')
    return zpe



def get_entropies(Frame):
    Entropies = df(columns=['E_pot', 'E_ZPE', 'Cv_trans', 'Cv_rot', 'Cv_vib', 'C_vtoC_p', 'S_trans',
        'S_rot', 'S_elec', 'S_vib', 'Sbar', 'S'], index=Frame.index)
    Entropies['E_pot'],Entropies['E_ZPE'],Entropies['Cv_trans'],Entropies['Cv_rot'],Entropies['Cv_vib'],Entropies['C_vtoC_p'],Entropies['S_trans'],Entropies['S_rot'],Entropies['S_elec'],Entropies['S_vib'],Entropies['Sbar'],Entropies['S']= zip(*Frame.apply(get_zpe_entropies, axis=1))
    try:
        Frame = Frame.join(Entropies)
    except:    
        Frame.update(Entropies)

    return Frame

def Entropylines(fp):
    Entropies={}
    with open(fp, "r") as f:
        try:
            for line in f :
                if line.startswith('E_pot'): #Loop until it finds E_pot then continues the next condition
                    Entropies['E_pot'] = float(line.split()[1] )
                    line = next(f)
                else:
                    continue

                if line.startswith('E_ZPE'): #Loop until it finds E_pot then continues the next condition
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
        except:
            return Entropies
        

#Entropy

def get_zpe_entropies(row):
    fp = row.root +row['Path']
    try:
        zpe =Zeropointenergy(fp+'/vib.out')
        entropies = Entropylines(fp+'/vib.out')
    except:
        zpe =Zeropointenergy(fp+'/out.txt')
        entropies = Entropylines(fp+'/out.txt')
    if len(entropies) == 12:
        return tuple(entropies.values())
    else:
        print(fp, '\n', entropies)
        return [NaN]*12
    



def gas_free_G(row, T=None):
    from sympy import symbols
    kb=8.617333262145E-5
    Temp=symbols('Temp')
    E = row['E']
    ZPE = row['E_ZPE']
    Cv_trans = row['Cv_trans']
    Cv_rot = row['Cv_rot']
    Cv_vib = row['Cv_vib']
    S_trans = row['S_trans']
    S_rot = row['S_rot']
    S_vib = row['S_vib']
    G = float(E)+float(ZPE)+Cv_trans+Cv_rot+Cv_vib-kb*Temp*(S_trans+S_rot+S_vib)
    if T==None:
        return G
    else:
        return G.subs('Temp', T)
    


def Atommultiindex(Frame):
    def getsymbols(row):
        struc = row['struc']
        Name = row.Name
        symbols = struc.get_chemical_symbols()
        names = [Name]*len(struc)
        indices = [n for n in np.arange(0,len(struc))]
        return names, symbols, indices

    nam, sym, ind = zip(*Frame.apply(getsymbols, axis=1))

    names = [i for n in nam for i in n]
    symbols = [i for n in sym for i in n]
    indices= [i for n in ind for i in n]
    #out = pd.MultiIndex(legel=[nam, ind])
    out = DataFrame(symbols, index=[names,indices], columns=['Symbols'])
    out.index.rename = ['Name', 'indices']
    return out

    


