#! /usr/bin/env python


from numpy import append
import warnings
from numpy import ComplexWarning
warnings.filterwarnings("ignore", category=ComplexWarning)


import ase
from ase.lattice import bulk
from ase import Atoms
from ase.constraints import FixAtoms, FixBondLength
from ase.build import surface
from ase.build import fcc111, fcc211, fcc110,fcc100

from ase.io import *
import os
import sys
from ase.calculators.vasp import Vasp
import ase.calculators.vasp as vasp_calculator
from ase.optimize import QuasiNewton
from ase.optimize import BFGS
import numpy as np
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo

atoms=read('relax.traj')

if 'Ni' in atoms.symbols:
    SPIN=2
else:
    SPIN=1

try:
  submitdir=sys.argv[1]
except:
  submitdir=''
if submitdir != '':
  submitdir += '/'

calc = vasp_calculator.Vasp(encut=500,
                        xc='PBE',
                        gga='BF',
                        luse_vdw=True,
                        zab_vdw=-1.8867,
                        kpts  = (6,6,1),
                        gamma = True, # Gamma-centered (defaults to Monkhorst-Pack)
                        ismear=1,
                        nelm=450,
                        lcharg=False,
                        algo = 'fast',
                        sigma = 0.1,
                        ibrion=-1,
                        ediffg=-0.01,  # forces
                        ediff=1e-7,  #energy conv.
                        lreal=False,
                        prec='normal',
                        istart=1,
                        nsw=50, # don't use the VASP internal relaxation, only use ASE
                        ispin=SPIN)

trajfile='final.traj'

path=sys.argv[1]
if 'CO2' in path:
    geometry='nonlinear'
    n=3
elif 'HCOO' in path:
    geometry='nonlinear'
    n=4
elif 'COOH' in path:
    geometry='nonlinear'
    n=1
elif '/CO/' in path:
    geometry='linear'
    n=2
elif '/H/' in path:
    geometry='monoatomic'
    n=1

indices = []
for i in ['C', 'O', 'H', 'K', 'Na']:
    try:
        indices = append(indices, atoms.symbols.indices()[i])
    except:
        print('no '+i+' in here')
indices = list(map(int,indices))

potentialenergy = atoms.get_potential_energy()
magmoms = [1 if i.symbol == 'Ni' else 0 for i in atoms]
atoms.set_initial_magnetic_moments(magmoms)
atoms.set_calculator(calc)
#magmoms = [1 if i.symbol == 'Ni' else 0 for i in atoms]
#atoms.set_initial_magnetic_moments(magmoms)

vib=Vibrations(atoms, indices=list(indices))
print(indices)

vib=Vibrations(atoms, indices=list(indices))
for n in os.listdir():
    if n.startswith('vib.'):
        if n.endswith('.pckl'):
            try:
                vib.read()
                print('vib read')
                break
            except:
                print('could not read')
                break
        print('no vib.pckl to read')
        break

vib.run()
vib.summary()
vib.write_jmol()
vib_energies = vib.get_energies()
thermo = IdealGasThermo(vib_energies=vib_energies,
                        potentialenergy=potentialenergy,
                        atoms=atoms,
                        geometry='nonlinear',
                        symmetrynumber=1, spin=SPIN-1)
G = thermo.get_gibbs_energy(temperature=500.15, pressure=101325.)
print('Gibbs energy',G)
f=open('gibbs.e','w')
f.write(str(G)+'\n')
f.close()
