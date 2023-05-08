import os
import pandas as pd
from ase.io import read

# class MyDataFrame:
#     def __init__(self, root):
#         self.root = root
#         self.df = pd.DataFrame(columns=['Path', 'Name', 'E', 'final_traj', 'OUTCAR', 'timestamp'])
#         self.update()

#     def update(self):
#         for path, dirs, files in os.walk(self.root):
#             if 'out.txt' in files:
#                 name = os.path.basename(path)
#                 idx = path.replace(self.root, '').replace('/', '-')
#                 out_path = os.path.join(path, 'out.txt')
#                 mod_time = os.path.getmtime(out_path)
#                 if idx in self.df.index and mod_time <= self.df.loc[idx, 'timestamp']:
#                     continue
#                 try:
#                     final_traj = read(os.path.join(path, 'final_traj'))
#                     energy = final_traj.get_potential_energy()
#                 except:
#                     final_traj = None
#                     energy = None
#                 if energy is None:
#                     try:
#                         outcar = read(os.path.join(path, 'OUTCAR'))
#                         energy = outcar.get_potential_energy()
#                     except:
#                         outcar = None
#                         energy = None
#                 if energy is not None:
#                     self.df.loc[idx] = [path.replace(self.root, ''), name, energy, final_traj, outcar, mod_time]

# import os
# from datetime import datetime
# from ase.io import read
# import pandas as pd

# class MyDataFrame:
#     def __init__(self, root):
#         self.root = root
#         self.data = self._create_dataframe()

#     def _create_dataframe(self):
#         rows = []
#         for dirpath, dirnames, filenames in os.walk(self.root):
#             if "out.txt" in filenames:
#                 path = dirpath.replace(self.root, "")
#                 name = os.path.basename(dirpath)
#                 index = path.replace("/", "-")
#                 final_traj_path = os.path.join(dirpath, "final.traj")
#                 outcar_path = os.path.join(dirpath, "OUTCAR")
#                 try:
#                     final_traj = read(final_traj_path)
#                 except:
#                     final_traj = None
#                 try:
#                     outcar = read(outcar_path)
#                 except:
#                     outcar = None
#                 timestamp = datetime.fromtimestamp(os.path.getmtime(os.path.join(dirpath, "out.txt")))
#                 e = None
#                 if final_traj is not None:
#                     e = final_traj.get_potential_energy()
#                 if e is None and outcar is not None:
#                     e = outcar.get_potential_energy()
#                 if e is not None:
#                     rows.append({"Name": name, "Path": path, "E final_traj": e, "OUTCAR": outcar, "timestamp": timestamp})
#         df = pd.DataFrame(rows)
#         df.set_index("Path", inplace=True)
#         return df

#     def update(self):
#         for path, row in self.data.iterrows():
#             out_path = os.path.join(self.root, path, "out.txt")
#             if os.path.exists(out_path):
#                 out_timestamp = datetime.fromtimestamp(os.path.getmtime(out_path))
#                 if out_timestamp > row["timestamp"]:
#                     self.data.loc[path, "timestamp"] = out_timestamp
#                     final_traj_path = os.path.join(self.root, path, "final.traj")
#                     outcar_path = os.path.join(self.root, path, "OUTCAR")
#                     try:
#                         final_traj = read(final_traj_path)
#                     except:
#                         final_traj = None
#                     try:
#                         outcar = read(outcar_path)
#                     except:
#                         outcar = None
#                     e = None
#                     if final_traj is not None:
#                         e = final_traj.get_potential_energy(default=None)
#                     if e is None and outcar is not None:
#                         e = outcar.get_potential_energy(default=None)
#                     if e is not None:
#                         self.data.loc[path, "E final_traj"] = e
#                         self.data.loc[path, "OUTCAR"] = outcar



import os
from ase.io import read
import pandas as pd
import numpy as np




class GeometryOptimizationDataFrame:
    def __init__(self, root):
        self.root = root
        self.data = self._create_dataframe()

    def _create_dataframe(self):
        data = []
        for path, dirs, files in os.walk(self.root):
            if 'out.txt' in files:
                try:
                    final_traj = read(os.path.join(path, 'final.traj'))
                except:
                    final_traj = None
                try:
                    outcar = read(os.path.join(path, 'OUTCAR'))
                except:
                    outcar = None
                if final_traj is None and outcar is None:
                    continue
                try:
                    energy = final_traj.get_potential_energy()
                    struc= final_traj
                except:
                    try:
                        energy = outcar.get_potential_energy()
                        struc = outcar
                    except:
                        energy = None
                timestamp = os.path.getmtime(os.path.join(path, 'out.txt'))

                Path = path.replace(self.root, '')
                name = Path.replace('/', '-')
                fmax =round(np.max(np.abs(struc.get_forces())), 4)
                
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

                data.append({'Name': name, 'Path': Path, 'files': os.listdir(os.path.join(self.root, path)),
                             'E': energy, 'struc': struc, 'final_traj': final_traj, 'OUTCAR': outcar, 'fmax':fmax, 'timestamp': timestamp, 'a' : a,  'b' : b,  'c' : c,  'gamma' : gamma, 'Formula' : Formula})            
                    

        return pd.DataFrame(data).set_index('Name')

    def update(self):
        self.data['new_timestamp'] = self.data.apply(lambda row: os.path.getmtime(self.root + row.name + '/out.txt'), axis=1)
        mask = self.data['new_timestamp'] > self.data['timestamp']
        for path in self.data[mask].index:
            try:
                final_traj = read(self.root + path + 'final.traj')
            except:
                final_traj = None
            try:
                outcar = read(self.root + path + 'OUTCAR')
            except:
                outcar = None
            if final_traj is None and outcar is None:
                continue
            try:
                energy = final_traj.get_potential_energy()
                struc = final_traj
            except:
                try:
                    energy = outcar.get_potential_energy()
                    struc = final_traj
                except:
                    energy = None

            self.data.loc[self.index, 'a'] = struc.cell.cellpar()[0]
            self.data.loc[self.index, 'b'] = struc.cell.cellpar()[1]
            self.data.loc[self.index, 'c'] = struc.cell.cellpar()[2]
            self.data.loc[self.index, 'gamma'] = struc.cell.cellpar()[5]

            self.data.loc[self.index, 'positions'] =struc.get_positions()
            #print(np.max(positions, axis=0)[2] , np.min(positions, axis=0)[2])
            self.data.loc[self.index, 'Formula'] = (''.join(set(struc.get_chemical_symbols()))
                    .replace('ZnCu', 'CuZn')
                    .replace('ZnNi', 'NiZn')
                    .replace('GaNi', 'NiGa')
            )

            self.data.loc[path, 'E'] = energy
            self.data.loc[path, 'final_traj'] = final_traj
            self.df.at[path, 'Files'] = os.listdir(os.path.join(self.root, path))
            self.data.loc[path, 'OUTCAR'] = outcar
            self.data.loc[path, 'timestamp'] = self.data.loc[path, 'new_timestamp']

            self.data.drop('new_timestamp', axis=1, inplace=True)


 #   def InputParameters(row):
 #       try:
 #           return row['final_traj'].calc.parameters
 #       except:           
 #           return {}


    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __delitem__(self, key):
        self.data.__delitem__(key)

    def __getattr__(self, attr):
        return getattr(self.data, attr)

    def __setattr__(self, attr, value):
        if attr in ['data', 'root']:
            super().__setattr__(attr, value)
        else:
            setattr(self.data, attr, value)

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()


#class Gas(GeometryOptimizationDataFrame):
#    def __init__(self, root):        
#        super().__init__(root)


class Surface(GeometryOptimizationDataFrame):

    def __init__(self, root):
        super().__init__(root)

        #self.Area = self.a*self.b*np.sin(np.radians(self.gamma))
        #self.vac = self.c-np.max(self.positions, axis=0)[2] + np.min(self.positions, axis=0)[2]#

        self.data.loc[self.index, 'B_clean']  = [True if 'clean' in i else False for i in self.index]
        self.data.loc[self.index, 'B_convergence']  = [True if 'convergence' in i else False for i in self.index]
        self.data.loc[self.index, 'B_adsorbate']  = [True if 'NH3' in i else False for i in self.index]
 
        for i in ['CHO', 'NH3', 'NH2_H' 'CuO', 'NiO', 'H', 'Hx4', 'O', 'Ox2', 'Ox3', 'OH', 'OHx2', 'HCOO', 'COOH', 'CO', 'CO2', 'COx2', 'Hx2', 'CN', 'CNx2']:
            if '-'+i+'-' in self.index:
                self.data.loc[self.index, 'B_adsorbate']  = True
                continue


class Adsorption(GeometryOptimizationDataFrame):

    def __init__(self, root, Adsorbates=['CHO', 'NH3', 'NH2_H' 'CuO', 'NiO', 'H', 'Hx4', 'O', 'Ox2', 'Ox3', 'OH', 'OHx2', 'HCOO', 'COOH', 'CO', 'CO2', 'COx2', 'Hx2', 'CN', 'CNx2']):
        super().__init__(root)
        
        #self.Area = self.a*self.b*np.sin(np.radians(self.gamma))
        #self.vac = self.c-np.max(self.positions, axis=0)[2] + np.min(self.positions, axis=0)[2]#

        self.data.loc[self.index, 'B_clean']  = [True if 'clean' in i else False for i in self.index]
        self.data.loc[self.index, 'B_adsorbate']  = [True if 'NH3' in i else False for i in self.index]
 
        for i in ['CHO', 'NH3', 'NH2_H' 'CuO', 'NiO', 'H', 'Hx4', 'O', 'Ox2', 'Ox3', 'OH', 'OHx2', 'HCOO', 'COOH', 'CO', 'CO2', 'COx2', 'Hx2', 'CN', 'CNx2']:
            if '-'+i+'-' in self.index:
                self.data.loc[self.index, 'B_adsorbate']  = True
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

    #self.ML = self.apply(lambda row: row["Ni"] + row["Zn"], axis=1)   
    #self.ML = self.apply(getML, axis=1)


class SurfaceAlloy(Surface):

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
            
    def __init__(self, root):
        self.ML = self.apply(getML, axis=1)


