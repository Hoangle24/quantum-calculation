# -*- coding: utf-8 -*-
"""
Created on Tue Oct 8:37:22 2021

@author: Hoang Le
"""

import re
import os
import csv
import glob 
import math
import shutil
import pandas
import numpy as np
import matplotlib.pyplot as plt  
from pathlib import Path as pt
from numpy.linalg import multi_dot
import subprocess as sp

ZERO = r"0\.?0*D?."
hatree = 27.2107
match_range = 2
NWchem = os.path.abspath(os.curdir)

def load_data (file_name):
    '''
    file_name (string): the name of the file containing the coordinates of the molecules (.xyz)
    Return: the coordinate of the orginal molecule that is easy to manipulate.
    each row of the final data is in the form of ['Z','x','y','z'] where Z is the atom and x,y, and z are the coordinates 
    '''
    inFile = open (file_name, 'r')
    file_line = inFile.readlines()
    coordinate = []
    for line in file_line:
        split_line = line.split ()
        if len (split_line) == 4:
            try: 
                if all(isinstance(float(x),float) for x in split_line [1:4]):
                    coordinate.extend ([split_line])
            except:
                pass
    return coordinate

class starting_molecule: 
    def __init__ (self,File_name): 
        self.coordinate = load_data (File_name)
    def get_coordinate(self):
        duplicate = self.coordinate.copy()
        return np.array(duplicate)
    def rotate(self,theta, phi=0, gamma=0):
        '''
        Used to rotate the molecule around the x,y,z-axis
        Input:
            theta (degree): rotation about the x-axis 
            phi (degree): rotation about the y-axis (default value = 0)
            gamma (degree): rotation about the z-axis (default value = 0)
        Return: the new coordinates of the rotated molecule.
        '''
        [theta,phi,gamma] = np.array([theta,phi,gamma])*np.pi/180#convert to radians
        original_molecule = self.get_coordinate()
        
        #Rotation matrices: Rx, Ry, and Rz
        Rx = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
        Ry = np.array([[np.cos(phi),0,np.sin(phi)],[0,1,0],[-np.sin(phi),0,np.cos(phi)]])
        Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],[np.sin(gamma),np.cos(gamma),0],[0,0,1]])
        R = multi_dot([Rz,Ry,Rx])

        for i,row in enumerate(R): 
            for j,index in enumerate (row):
                if abs (index) < 0.0000001:
                    R[i,j] = 0
        for i in range (0,len(original_molecule)):
            x = float(original_molecule[i][1])
            y = float(original_molecule [i][2])
            z = float(original_molecule [i][3])
            [x1,y1,z1] = R.dot(np.array([x,y,z]))
            original_molecule[i][1] = format(x1,'.10f')
            original_molecule[i][2] = format(y1,'.10f')
            original_molecule[i][3] = format(z1,'.10f')   
        return np.array(original_molecule)
    
def shift_xyz(co1,x,y,z):
    co = co1.copy()
    for i in range (0,len (co)):
        co[i][1] = str(float (co[i][1]) +x)
        co[i][2] = str(float (co[i][2]) +y)
        co[i][3] = str(float (co[i][3]) +z)
    return np.array(co) 

def write_file_nw (name,m0,m1,x,y,z):
    file_name = '{}.nw'.format(name)
    molecule = re.search(r'(?P<mol>\w*_x\d*\.\d*_y\d*\.\d*_z\d*\.\d*(_theta\d*\.?\d*)?(_phi\d*\.?\d*)?(_gamma\d*\.?\d*)?)\.nw',file_name).group('mol')
    f = open (file_name,'w')
    f.write('echo\n')
    f.write(' '.join(['start',molecule,'\n']))
    f.write(' '.join(['title','"B3LYP/6-31G*',molecule,'"','\n']))
    f.write('charge 0\ngeometry units angstroms print xyz noautoz noautosym')
    for i in range (0,len(m0)):
        f.write ('\n  ')
        f.write ('        '.join(m0[i]))
    new_m1 = shift_xyz (m1,x,y,z)
    for i in range (0, len(new_m1)):
        f.write ('\n  ')
        f.write ('        '.join(new_m1[i]))
    f.write('\nend\n\nbasis\n  * library 6-31G*\nend\n\n')
    f.write('dft\n  xc b3lyp\n  mult 1\n  disp vdw 3\nend\n\n')
    f.write('tddft\n  nroots 10\nend\n\nset grid:eaf_size_in_dbl              8304257\n\ntask tddft energy')
    f.close()
    
def write_file_xyz (name,m0,m1,x,y,z):
    file_name = '{}.xyz'.format(name)
    f = open (file_name,'w')
    f.write(str(len(m0)+len(m1)))
    f.write('\n')
    for i in range (0,len(m0)):
        f.write ('\n  ')
        f.write ('        '.join(m0[i]))
    new_m1 = shift_xyz (m1,x,y,z)
    for i in range (0, len(new_m1)):
        f.write ('\n  ')
        f.write ('        '.join(new_m1[i]))
    f.close()

def numerate (string):
    num = float(re.search(r'[+-]?\d*\.\d+',string).group())
    try:
        expo= float(re.search('(?<=[DEe])[+-]?\d+',string).group())
    except:
        expo= 0
    return num*10**expo
   
def load_mo_data(file_name, MATCH_RANGE = match_range):  
    '''This function will scan through a file and look for the pattern of the molecular orbital 
    data and return a list of related HOMO and LUMO energy levels.
    Input: 
        file_name (string):targeted file's name 
        MATCH_RANGE: how far off the HUMO LUMO level you want to look at
    Return: list of HUMO and LUMO energy levels.'''
    pattern = r"Vector\s*(\d*)\s*Occ=(?P<Occ>\S*)\s*E=(?P<E>\s*?[+-]?\d*\.\d+D[+-]?\d+)\s*(Symmetry=(\w))?"
    with open(file_name) as f:
        data = f.read()
    matches = list(re.finditer(pattern, data))
    homo_lumo = []
    for i, match in enumerate(matches):
        if bool(re.match(ZERO, match.group('Occ'))): 
            e_data = matches[i - MATCH_RANGE: i + MATCH_RANGE]
            for count,item in enumerate(e_data): 
                homo_lumo.append(numerate(item.group('E'))*hatree)
            return homo_lumo

def load_excited_states (file_name):
    '''This function will look for the pattern of excited states' data from the targeted file
    Input:
        file_name (string):targeted file's name 
    Return: list of excited states and their corresponding oscillator strength (absorbtion probability)'''
    root_pattern = r"(?P<lv>Root\s*\d*)\s*singlet a.\s*(?P<au>[+-]?\d*\.\d+)\s*a\.u\.\s*(?P<E>[+-]?\d*\.\d+)\s*eV"
    osc_pattern = r"Total Oscillator Strength\s*(?P<osc>\d*\.\d+)"
    with open (file_name) as f: 
        data = f.read()
    matches = list(re.finditer(root_pattern, data))
    lim = [i.span() for i in matches]
    excited_data = [] #     :D
    for i, match in enumerate(matches):
        try:
            osc_strength = numerate(re.search(osc_pattern, data[lim[i][1]:lim[i+1][0]]).group('osc'))
        except:
            osc_strength = numerate(re.search(osc_pattern, data[lim[i][1]:]).group('osc'))
        excited_data.append([numerate(match.group('E')),osc_strength])
    return excited_data

def transition_mo (file_name,exc_states = match_range):
    root_pattern = r"(?P<lv>Root\s*\d*)\s*singlet a.\s*(?P<au>[+-]?\d*\.\d+)\s*a\.u\.\s*(?P<E>[+-]?\d*\.\d+)\s*eV"
    occ_pattern = r"\s*Occ\.\s*(?P<Occ>\d*)\s*a?\s*-*\s*Virt\.\s*(?P<Virt>\d*)\s*a?\s*(?P<vec>[+-]?\d*\.\d*)\s*[XY]?"
    with open (file_name) as f: 
        data = f.read()
    matches = list(re.finditer(root_pattern, data))
    lim = [i.span() for i in matches]
    orbital = []
    if lim != []:
        for i in range(exc_states):
            try:
                transvec = list (re.finditer(occ_pattern, data[lim[i][1]:lim[i+1][0]]))
            except:
                transvec = list (re.finditer(occ_pattern, data[lim[i][1]:]))
            vec =[numerate(x.group('vec')) for x in transvec]
            max_vec = max (vec)
            max_index = vec.index (max_vec)
            orb = [transvec[max_index].group(name) for name in ['Occ','Virt']]+[max_vec]
            orbital.append (orb)
        return orbital
    else: 
        return []
    
def extract_transmo (folder,exc_num=10):
    '''This function extracts the data from all log files in the input folder'''
    log_path = os.path.join (folder,'*.log')
    log_files = glob.glob (log_path)
    pattern = r'(?P<mol>\w*-?\w*)_x(?P<xdis>\d*\.\d+)_y(?P<ydis>\d*\.\d+)_z(?P<zdis>\d*\.\d+).log'
    numfiles = len (log_files)
    xdis = np.zeros(numfiles)
    ydis = np.zeros(numfiles)
    zdis = np.zeros(numfiles)
    occ = np.zeros([match_range,numfiles])
    virt = np.zeros([match_range,numfiles])
    vec = np.zeros([match_range,numfiles]) 
    
    for i,file in enumerate(log_files):
        x = re.search (pattern,file).group('xdis')
        y = re.search (pattern,file).group('ydis')
        z = re.search (pattern,file).group('zdis')
        molecule = re.search (pattern,file).group('mol')
        o = [k[0] for k in transition_mo(file)]
        if o == []:
            print ('Potential error in file: {}_x{}_y{}_z{}.log'.format (molecule,x,y,z))
        else:
            v = [k[1] for k in transition_mo(file)]
            ve =  [k[2] for k in transition_mo(file)]
            occ[:, i]= o 
            virt[:,i]=v
            vec[:,i]=ve
        xdis[i] = x
        ydis[i] = y 
        zdis[i] = z 
        
    return (xdis,ydis,zdis,occ,virt,vec)

def transmo_process (greedyfolder):
    '''
    This function will process all calculated .log files in any subfolders in the directory: /pscratch/anthony_uksr/hmle231/NWchem/calculation_data
    and generate a csv file containing the excited state transition data (the MOs with the highest probability of participating in a transition and their corresponding values. 
    Input:
        greddyfolder (string): Name of the folder containing the log files in /pscratch/anthony_uksr/hmle231/NWchem/calculation_data/
        exc_num(integer): Number of excited states (default=10. DO NOT CHANGE UNLESS NECESSARY)
    Return:
        
    '''
    folder_pattern = r"%s(_\d*\.?\d*(-\d*\.?\d*)?){3}"%greedyfolder
    NWchem = os.path.abspath(os.curdir)
    folder_path = os.path.join(NWchem,'calculation_data','*')
    folders = glob.glob (folder_path)
    xdis,ydis,zdis=(np.array([]),)*3
    occ,virt,vec = (np.empty([match_range,1]),)*3
    for subfolder in folders: 
        if re.search (folder_pattern,subfolder) != None:
            (x,y,z,o,v,ve) = extract_transmo (subfolder)
            xdis = np.concatenate((xdis,x),axis=0)
            ydis = np.concatenate((ydis,y),axis=0)
            zdis = np.concatenate((zdis,z),axis=0)
            occ = np.concatenate((occ,o),axis=1)
            virt = np.concatenate((virt,v),axis=1)
            vec = np.concatenate((vec,ve),axis=1)
    occ = np.delete(occ,0,1) 
    virt=np.delete (virt,0,1)
    vec=np.delete (vec,0,1)
      
    #################################################################
    #####   Writing the csv file from the extracted data
    ##  HEADER
    NAME = [j.format(i) for i in range(1,1+match_range) for j in ['Occ{}','Virt{}','Vec{}'] ]
    HEADER = ['x-dis','y-dis','z-dis']+NAME
    ##File_write
    saved_file = os.path.join(NWchem,'processed_csv','Transition_vector_{}.csv'.format(greedyfolder))
    #write data 
    with open (saved_file,'w', newline='') as csvfile:
        writer = csv.writer (csvfile)
        writer.writerow (HEADER)
        for i in range(len(xdis)):
            if occ[1][i]==0 and virt[1][i]==0:
                pass
            else:
                writer.writerow ([xdis[i],ydis[i],zdis[i]]+
                                 [item for sublist in [[occ[j][i],virt[j][i],vec[j][i]] for j in range (match_range)] for item in sublist])
    sortfile = pandas.read_csv(saved_file)
    #sorting xyz 
    newfile = sortfile.sort_values (['x-dis','y-dis','z-dis']).to_csv(index=False)
    with open (saved_file,'w',newline='') as nsfile:
        nsfile.write(newfile) 

def extract_folder_data (folder,exc_num=10):
    '''This function extracts the data from all log files in the input folder'''
    log_path = os.path.join (folder,'*.log')
    log_files = glob.glob (log_path)
    pattern = r'(?P<mol>\w*-?\w*)_x(?P<xdis>\d*\.\d+)_y(?P<ydis>\d*\.\d+)_z(?P<zdis>\d*\.\d+)(_theta\d*\.?\d*)?.log'
    numfiles = len (log_files)
    xdis = np.zeros(numfiles)
    ydis = np.zeros(numfiles)
    zdis = np.zeros(numfiles)
    mo = np.zeros ([match_range*2, numfiles])
    excited_states = np.zeros([exc_num,numfiles])
    osc_strength =  np.zeros([exc_num,numfiles])
    error_files = []  
    for i,file in enumerate(log_files):
        x = re.search (pattern,file).group('xdis')
        y = re.search (pattern,file).group('ydis')
        z = re.search (pattern,file).group('zdis')
        molecule = re.search (pattern,file).group('mol')
        holu = load_mo_data (file)
        exc =  [k[0] for k in load_excited_states(file)]
        if exc == []:
            #print ('Potential error in file: {}_x{}_y{}_z{}.log'.format (molecule,x,y,z))
            error_files.append (i)
        else:
            osc = [k[1] for k in load_excited_states(file)]
            excited_states[:,i] = exc
            osc_strength[:,i] = osc
        mo[:,i] = holu
        xdis[i] = x
        ydis[i] = y 
        zdis[i] = z 
    return (xdis,ydis,zdis,mo,excited_states,osc_strength)

def log_process (Folder,exc_num=10):
    '''
    This function will process all calculated .log files in any subfolders in the directory: /pscratch/anthony_uksr/hmle231/NWchem/calculation_data
    and generate a csv file containing the result in the folder /pscratch/anthony_uksr/hmle231/NWchem/processed_csv.
    Input:
        Foler(string): Name of the folder containing the log files in /pscratch/anthony_uksr/hmle231/NWchem/calculation_data/
        exc_num(integer): Number of excited states (default=10. DO NOT CHANGE UNLESS NECESSARY)
    Return:
        
    '''
    #create a folder path and search pattern for lag files
    NWchem = os.path.abspath(os.curdir)
    folder_path = os.path.join (NWchem,'calculation_data',Folder,'*.log')
    files = glob.glob (folder_path)
    pattern = r'(?P<mol>\w*-?\w*)_x(?P<xdis>\d*\.\d+)_y(?P<ydis>\d*\.\d+)_z(?P<zdis>\d*\.\d+).log'
    #create matrices to store the data
    numfiles = len (files)
    xdis = np.zeros(numfiles)
    ydis = np.zeros(numfiles)
    zdis = np.zeros(numfiles)
    mo = np.zeros ([match_range*2, numfiles])
    excited_states = np.zeros([exc_num,numfiles])
    osc_strength =  np.zeros([exc_num,numfiles])
    error_files = []   
    
    #extract the data from the log files 
    for i,file in enumerate(files):
        x = re.search (pattern,file).group('xdis')
        y = re.search (pattern,file).group('ydis')
        z = re.search (pattern,file).group('zdis')
        molecule = re.search (pattern,file).group('mol')
        filename = re.search (pattern, file).group()
        logfile = os.path.join (NWchem,'calculation_data',Folder,filename)
        holu = load_mo_data (logfile)
        exc =  [k[0] for k in load_excited_states(logfile)]
        if exc == []:
            print ('Potential error in file: {}_x{}_y{}_z{}.log'.format (molecule,x,y,z))
            error_files.append (i)
        else:
            osc = [k[1] for k in load_excited_states(logfile)]
            excited_states[:,i] = exc
            osc_strength[:,i] = osc
        mo[:,i] = holu
        xdis[i] = x
        ydis[i] = y 
        zdis[i] = z 
    
    #################################################################
    #####   Writing the csv file from the extracted data
    ##  HEADER
    NAME = [i for i in range (2*match_range)]
    for i in range (0,match_range):
        if i == 0:
            NAME[match_range-1] ='HOMO'
            NAME[-match_range] ='LUMO'
        else:
            NAME[match_range-1-i]='HOMO-'+str(i)
            NAME[-match_range+i]='LUMO+'+str(i)
    HEADER = ['x-dis','y-dis','z-dis']+NAME
    for i in range (1,11):
        HEADER = HEADER + ['EXC{}'.format(i)]
    for i in range (1,11):
        HEADER = HEADER + ['PROB{}'.format(i)]
    ##File_write
    saved_file = os.path.join(NWchem,'processed_csv','{}_processed.csv'.format(Folder))
    with open (saved_file,'w', newline='') as csvfile:
        writer = csv.writer (csvfile)
        writer.writerow (HEADER)
        for i in range(numfiles):
            writer.writerow ([xdis[i],ydis[i],zdis[i]]+
                              [mo[j][i] for j in range(len(NAME))]+
                              [excited_states[j][i] for j in range(10)]+
                              [osc_strength[j][i] for j in range(10)])
    sortfile = pandas.read_csv(saved_file)
    newfile = sortfile.sort_values (['x-dis','y-dis','z-dis']).to_csv(index=False)
    with open (saved_file,'w',newline='') as nsfile:
        nsfile.write(newfile)
    #####################################################################
    excited_states=np.delete (excited_states,error_files,1)#remove the bad calculations 
    osc_strength=np.delete (osc_strength,error_files,1)#remove the bad calculations
    return {'distance':np.array([xdis,ydis,zdis]),'MO':mo, 'excited states':excited_states, 'absorption probability':osc_strength}

def greedyprocess (greedyfolder,exc_num=10):
    '''
    This function will process all calculated .log files in any subfolders in the directory: /pscratch/anthony_uksr/hmle231/NWchem/calculation_data
    and generate a csv file containing the result in the folder /pscratch/anthony_uksr/hmle231/NWchem/processed_csv.
    Input:
        Foler(string): Name of the folder containing the log files in /pscratch/anthony_uksr/hmle231/NWchem/calculation_data/
        exc_num(integer): Number of excited states (default=10. DO NOT CHANGE UNLESS NECESSARY)
    Return:
        
    '''
    folder_pattern = r"%s(_\d*\.?\d*(-\d*\.?\d*)?){3}"%greedyfolder
    NWchem = os.path.abspath(os.curdir)
    folder_path = os.path.join(NWchem,'calculation_data','*')
    folders = glob.glob (folder_path)
    xdis,ydis,zdis=(np.array([]),)*3
    MO = np.empty([match_range*2,1])
    excited_states,osc_strength=(np.empty([exc_num,1]),)*2
    for subfolder in folders: 
        if re.search (folder_pattern,subfolder) != None:
            (x,y,z,mo,exc,osc) = extract_folder_data (subfolder)
            xdis = np.concatenate((xdis,x),axis=0)
            ydis = np.concatenate((ydis,y),axis=0)
            zdis = np.concatenate((zdis,z),axis=0)
            MO = np.concatenate((MO,mo),axis=1)
            excited_states = np.concatenate((excited_states,exc),axis=1)
            osc_strength = np.concatenate((osc_strength,osc),axis=1)
    MO = np.delete(MO,0,1) 
    excited_states=np.delete (excited_states,0,1)
    osc_strength=np.delete (osc_strength,0,1)
      
    #################################################################
    #####   Writing the csv file from the extracted data
    ##  HEADER
    NAME = [i for i in range (2*match_range)]
    for i in range (0,match_range):
        if i == 0:
            NAME[match_range-1] ='HOMO'
            NAME[-match_range] ='LUMO'
        else:
            NAME[match_range-1-i]='HOMO-'+str(i)
            NAME[-match_range+i]='LUMO+'+str(i)
    HEADER = ['x-dis','y-dis','z-dis']+NAME
    for i in range (1,11):
        HEADER = HEADER + ['EXC{}'.format(i)]
    for i in range (1,11):
        HEADER = HEADER + ['PROB{}'.format(i)]
    ##File_write
    saved_file = os.path.join(NWchem,'processed_csv','{}_processed.csv'.format(greedyfolder))
    with open (saved_file,'w', newline='') as csvfile:
        writer = csv.writer (csvfile)
        writer.writerow (HEADER)
        for i in range(len(xdis)):
            writer.writerow ([xdis[i],ydis[i],zdis[i]]+
                              [MO[j][i] for j in range(match_range*2)]+
                              [excited_states[j][i] for j in range(exc_num)]+
                              [osc_strength[j][i] for j in range(exc_num)])
    sortfile = pandas.read_csv(saved_file)
    newfile = sortfile.sort_values (['x-dis','y-dis','z-dis']).to_csv(index=False)
    with open (saved_file,'w',newline='') as nsfile:
        nsfile.write(newfile)
def core_partition (compute_cores):
    def inner (func):     
        def wrapper (*args,**kwargs):
            di = func (*args,*kwargs)        
            all_files = glob.glob (os.path.join(di,'*'))
            file_count = len (all_files)
            folder_capacity= math.ceil (file_count/compute_cores)
            files = os.listdir(di)
            for core in range(compute_cores):
                sub = 'core_{}'.format(core+1)
                subfolder = os.path.join (di,sub)
                if os.path.exists(subfolder):
                    shutil.rmtree(subfolder)
                os.mkdir(subfolder)
                for i in range (folder_capacity):
                    try:
                        f = all_files[core*folder_capacity+i] 
                        dest = os.path.join(subfolder,files[core*folder_capacity+i])                
                        os.rename(f,dest)
                    except:
                        pass
                sp.run('cd {}; sbatch $HOME/scripts/nwcbatch.sh'.format (os.path.join(di,sub)), shell=True)
        return wrapper 
    return inner 
 
@core_partition(10)    
def shifting_scan (File_name,Folder,xstart,xstop,ystart,ystop,zstart,zstop,space,FileType='nw',theta=0,phi=0,gamma=0):
    '''This function will generate all necessary nw or xyz files in the directory: /pscratch/anthony_uksr/hmle231/NWchem/calculation_data
    Input:
        File_name (string): The name of the original xyz file stored in the directory: /pscratch/anthony_uksr/hmle231/NWchem/original_xyz_file
        Folder (string): The name of the subfolder you want to store the files in
        xstart, xstop, ystart, ystop, zstart, zstop (float): The start and stop values of the coordinate scan(if the one axis scan is not necessary, set start=stop)
        space (float): The spacing between the scans
        FileType (string): the file type you want to generate (either 'nw' or 'xyz')
    Return:
        None 
    '''
    compound = re.search (r'(?P<mol>\w*-?\w*)\.xyz',File_name).group('mol')
    NWchem = os.path.abspath(os.curdir)
    di = os.path.join(NWchem,'calculation_data',Folder) #Folder directory  
    #create the Folder path
    if os.path.exists(di):
        shutil.rmtree(di)
    os.mkdir(di)
    #generate molecule coordinate 
    originalFilePath = os.path.join (NWchem,'original_xyz_file',File_name)
    molecule0 = starting_molecule(originalFilePath).get_coordinate()
    molecule1 = starting_molecule(originalFilePath).rotate(theta, phi, gamma)
    
    #create calculation files. 

    numx = math.floor ((xstop-xstart)/space)+1 
    numy = math.floor ((ystop-ystart)/space)+1 
    numz = math.floor ((zstop-zstart)/space)+1
    count = 0
    for i in range (numx):
        for j in range (numy):
            for k in range (numz):
                file_ij = '{}_x{}_y{}_z{}'.format(compound,round(xstart+space*i,3),round(ystart+space*j,3),round (zstart+space*k,3))
                file_ij = os.path.join(di,file_ij)
                if FileType == 'nw':
                    write_file_nw(file_ij,molecule0, molecule1,xstart+space*i,ystart+space*j,zstart+space*k)
                elif FileType == 'xyz':
                    write_file_xyz(file_ij,molecule0, molecule1,xstart+space*i,ystart+space*j,zstart+space*k)
                else:
                    raise NameError('FileType={}: name {} is not defined'.format(FileType,FileType))
                count += 1
    return di





@core_partition(10)
def rotation_scan (File_name,Folder,xstart,xstop,ystart,ystop,zstart,zstop,xyzspace,tstart, tstop, aspace, FileType='nw'):
    compound = re.search (r'(?P<mol>\w*-?\w*)\.xyz',File_name).group('mol')
    di = os.path.join(NWchem,'calculation_data',Folder) #Folder directory  
    #create the Folder path
    if os.path.exists(di):
        shutil.rmtree(di)
    os.mkdir(di)
    #generate molecule coordinate 
    originalFilePath = os.path.join (NWchem,'original_xyz_file',File_name)
    molecule0 = starting_molecule(originalFilePath).get_coordinate()
    def num (start,stop,space):
        return math.floor ((stop-start)/space)+1
    file_count = 0
    for i in range (num(tstart,tstop,aspace)):
        for j in range (num(xstart, xstop,xyzspace)):
            for k in range (num(ystart, ystop,xyzspace)):
                for l in range (num(zstart, zstop,xyzspace)):
                    molecule1 = starting_molecule(originalFilePath).rotate(tstart+aspace*i)
                    file_ij = '{}_x{}_y{}_z{}_theta{}'.format(compound,round(xstart+xyzspace*j,3),round(ystart+xyzspace*k,3),round (zstart+xyzspace*l,3),tstart+aspace*i)
                    file_ij = os.path.join(di,file_ij)
                    if FileType =='xyz':
                        write_file_xyz(file_ij,molecule0, molecule1,xstart+xyzspace*j,ystart+xyzspace*k,zstart+xyzspace*l+1.5*math.sin((tstart+aspace*i)*np.pi/180))
                    elif FileType == 'nw':
                        write_file_nw(file_ij,molecule0, molecule1,xstart+xyzspace*j,ystart+xyzspace*k,zstart+xyzspace*l+1.5*math.sin((tstart+aspace*i)*np.pi/180))
                    else:
                        raise NameError('FileType={}: unrecognized file format'.format(FileType,FileType))
                    file_count+=1
    return di



#sp.run ('cd /pscratch/anthony_uksr/hmle231/NWchem/calculation_data/{}/{}; sbatch $HOME/scripts/nwcbatch.sh'.format (Folder,sub), shell=True)
 #       print (os.listdir(subfolder))

def core_process_rotation (folder,exc_num=10):
    '''This function extracts the data from all log files in the input folder'''
    log_path = os.path.join (folder,'*.log')
    log_files = glob.glob (log_path)
    pattern = r'(?P<mol>\w*-?\w*)_x(?P<xdis>\d*\.\d+)_y(?P<ydis>\d*\.\d+)_z(?P<zdis>\d*\.\d+)_theta(?P<theta>\d*\.?\d*).log'
    numfiles = len (log_files)
    xdis = np.zeros(numfiles)
    ydis = np.zeros(numfiles)
    zdis = np.zeros(numfiles)
    theta = np.zeros(numfiles)
    mo = np.zeros ([match_range*2, numfiles])
    excited_states = np.zeros([exc_num,numfiles])
    osc_strength =  np.zeros([exc_num,numfiles])
    error_files = []  
    for i,file in enumerate(log_files):
        x,y,z,the,molecule = re.search (pattern,file).group('xdis','ydis','zdis','theta','mol')
        holu = load_mo_data (file)
        exc =  [k[0] for k in load_excited_states(file)]
        if exc == []:
            print ('Potential error in file: {}_x{}_y{}_z{}.log'.format (molecule,x,y,z))            
            error_files.append (i)
        else:
            osc = [k[1] for k in load_excited_states(file)]
            excited_states[:,i] = exc
            osc_strength[:,i] = osc
        #mo[:,i],xdis[i],ydis[i],zdis[i],theta[i] = (holu,x,y,z,the)
        theta[i] = the
        mo[:,i] = holu
        xdis[i] = x
        ydis[i] = y 
        zdis[i] = z 
    return (xdis,ydis,zdis,theta,mo,excited_states,osc_strength)

def main_folder_process (Folder,rotation=True,exc_num=10):
    '''
    This function will process all calculated .log files in any subfolders in the directory: /pscratch/anthony_uksr/hmle231/NWchem/calculation_data
    and generate a csv file containing the result in the folder /pscratch/anthony_uksr/hmle231/NWchem/processed_csv.
    Input:
        Foler(string): Name of the folder containing the log files in /pscratch/anthony_uksr/hmle231/NWchem/calculation_data/
        exc_num(integer): Number of excited states (default=10. DO NOT CHANGE UNLESS NECESSARY)
    Return:
    '''
    NWchem = os.path.abspath(os.curdir)
    folder_path = os.path.join(NWchem,'calculation_data',Folder,'*')
    folders = glob.glob (folder_path)
    xdis,ydis,zdis,theta=(np.array([]),)*4
    MO = np.empty([match_range*2,1])
    excited_states,osc_strength=(np.empty([exc_num,1]),)*2
    
    for subfolder in folders: 
        if rotation==True:
            (x,y,z,the,mo,exc,osc) = core_process_rotation (subfolder)
            theta = np.concatenate((theta,the),axis=0)
        else: 
            (x,y,z,mo,exc,osc) = extract_folder_data (subfolder)
            the = np.zeros(x.shape)
            theta = np.concatenate((theta,the),axis=0)
        xdis = np.concatenate((xdis,x),axis=0)
        ydis = np.concatenate((ydis,y),axis=0)
        zdis = np.concatenate((zdis,z),axis=0)
        MO = np.concatenate((MO,mo),axis=1)
        excited_states = np.concatenate((excited_states,exc),axis=1)
        osc_strength = np.concatenate((osc_strength,osc),axis=1)
    
    MO = np.delete(MO,0,1) 
    excited_states=np.delete (excited_states,0,1)
    osc_strength=np.delete (osc_strength,0,1)
      
    #################################################################
    #####   Writing the csv file from the extracted data
    ##  HEADER
    NAME = [i for i in range (2*match_range)]
    for i in range (0,match_range):
        if i == 0:
            NAME[match_range-1] ='HOMO'
            NAME[-match_range] ='LUMO'
        else:
            NAME[match_range-1-i]='HOMO-'+str(i)
            NAME[-match_range+i]='LUMO+'+str(i)
    HEADER = ['x-dis','y-dis','z-dis','theta']+NAME
    for i in range (1,11):
        HEADER = HEADER + ['EXC{}'.format(i)]
    for i in range (1,11):
        HEADER = HEADER + ['PROB{}'.format(i)]
    ##File_write
    saved_file = os.path.join(NWchem,'processed_csv','{}.csv'.format(Folder))
    with open (saved_file,'w', newline='') as csvfile:
        writer = csv.writer (csvfile)
        writer.writerow (HEADER)
        for i in range(len(xdis)):
            writer.writerow ([xdis[i],ydis[i],zdis[i],theta[i]]+
                              [MO[j][i] for j in range(match_range*2)]+
                              [excited_states[j][i] for j in range(exc_num)]+
                              [osc_strength[j][i] for j in range(exc_num)])
    sortfile = pandas.read_csv(saved_file)
    newfile = sortfile.sort_values (['x-dis','y-dis','z-dis','theta']).to_csv(index=False)
    with open (saved_file,'w',newline='') as nsfile:
        nsfile.write(newfile)
    return (MO)






###############################################################################################################################################
    #Temporary code for quick scan of gamma. Need modification later 

@core_partition(10)
def gamma_scan (File_name,Folder,xstart,xstop,ystart,ystop,zstart,zstop,xyzspace,tstart, tstop, aspace, FileType='nw'):
    compound = re.search (r'(?P<mol>\w*-?\w*)\.xyz',File_name).group('mol')
    di = os.path.join(NWchem,'calculation_data',Folder) #Folder directory  
    #create the Folder path
    if os.path.exists(di):
        shutil.rmtree(di)
    os.mkdir(di)
    #generate molecule coordinate 
    originalFilePath = os.path.join (NWchem,'original_xyz_file',File_name)
    molecule0 = starting_molecule(originalFilePath).get_coordinate()
    def num (start,stop,space):
        return math.floor ((stop-start)/space)+1
    file_count = 0
    for i in range (num(tstart,tstop,aspace)):
        for j in range (num(xstart, xstop,xyzspace)):
            for k in range (num(ystart, ystop,xyzspace)):
                for l in range (num(zstart, zstop,xyzspace)):
                    molecule1 = starting_molecule(originalFilePath).rotate(0,gamma=tstart+aspace*i)
                    file_ij = '{}_x{}_y{}_z{}_gamma{}'.format(compound,round(xstart+xyzspace*j,3),round(ystart+xyzspace*k,3),round (zstart+xyzspace*l,3),tstart+aspace*i)
                    file_ij = os.path.join(di,file_ij)
                    if FileType =='xyz':
                        write_file_xyz(file_ij,molecule0, molecule1,xstart+xyzspace*j,ystart+xyzspace*k,zstart+xyzspace*l+1.5*math.sin((tstart+aspace*i)*np.pi/180))
                    elif FileType == 'nw':
                        write_file_nw(file_ij,molecule0, molecule1,xstart+xyzspace*j,ystart+xyzspace*k,zstart+xyzspace*l+1.5*math.sin((tstart+aspace*i)*np.pi/180))
                    else:
                        raise NameError('FileType={}: unrecognized file format'.format(FileType,FileType))
                    file_count+=1
    return di


def core_process_gamma (folder,exc_num=10):
    '''This function extracts the data from all log files in the input folder'''
    log_path = os.path.join (folder,'*.log')
    log_files = glob.glob (log_path)
    pattern = r'(?P<mol>\w*-?\w*)_x(?P<xdis>\d*\.\d+)_y(?P<ydis>\d*\.\d+)_z(?P<zdis>\d*\.\d+)_gamma(?P<theta>\d*\.?\d*).log'
    numfiles = len (log_files)
    xdis = np.zeros(numfiles)
    ydis = np.zeros(numfiles)
    zdis = np.zeros(numfiles)
    theta = np.zeros(numfiles)
    mo = np.zeros ([match_range*2, numfiles])
    excited_states = np.zeros([exc_num,numfiles])
    osc_strength =  np.zeros([exc_num,numfiles])
    error_files = []  
    for i,file in enumerate(log_files):
        x,y,z,the,molecule = re.search (pattern,file).group('xdis','ydis','zdis','theta','mol')
        holu = load_mo_data (file)
        exc =  [k[0] for k in load_excited_states(file)]
        if exc == []:
            print ('Potential error in file: {}_x{}_y{}_z{}_gamma{}.log'.format (molecule,x,y,z,theta))            
            error_files.append (i)
        else:
            osc = [k[1] for k in load_excited_states(file)]
            excited_states[:,i] = exc
            osc_strength[:,i] = osc
        #mo[:,i],xdis[i],ydis[i],zdis[i],theta[i] = (holu,x,y,z,the)
        theta[i] = the
        mo[:,i] = holu
        xdis[i] = x
        ydis[i] = y 
        zdis[i] = z 
    return (xdis,ydis,zdis,theta,mo,excited_states,osc_strength)

def main_gamma_process (Folder,rotation=True,exc_num=10):
    '''
    This function will process all calculated .log files in any subfolders in the directory: /pscratch/anthony_uksr/hmle231/NWchem/calculation_data
    and generate a csv file containing the result in the folder /pscratch/anthony_uksr/hmle231/NWchem/processed_csv.
    Input:
        Foler(string): Name of the folder containing the log files in /pscratch/anthony_uksr/hmle231/NWchem/calculation_data/
        exc_num(integer): Number of excited states (default=10. DO NOT CHANGE UNLESS NECESSARY)
    Return:
    '''
    NWchem = os.path.abspath(os.curdir)
    folder_path = os.path.join(NWchem,'calculation_data',Folder,'*')
    folders = glob.glob (folder_path)
    xdis,ydis,zdis,theta=(np.array([]),)*4
    MO = np.empty([match_range*2,1])
    excited_states,osc_strength=(np.empty([exc_num,1]),)*2
    
    for subfolder in folders: 
        if rotation==True:
            (x,y,z,the,mo,exc,osc) = core_process_gamma (subfolder)
            theta = np.concatenate((theta,the),axis=0)
        else: 
            (x,y,z,mo,exc,osc) = extract_folder_data (subfolder)
            the = np.zeros(x.shape)
            theta = np.concatenate((theta,the),axis=0)
        xdis = np.concatenate((xdis,x),axis=0)
        ydis = np.concatenate((ydis,y),axis=0)
        zdis = np.concatenate((zdis,z),axis=0)
        MO = np.concatenate((MO,mo),axis=1)
        excited_states = np.concatenate((excited_states,exc),axis=1)
        osc_strength = np.concatenate((osc_strength,osc),axis=1)
    
    MO = np.delete(MO,0,1) 
    excited_states=np.delete (excited_states,0,1)
    osc_strength=np.delete (osc_strength,0,1)
      
    #################################################################
    #####   Writing the csv file from the extracted data
    ##  HEADER
    NAME = [i for i in range (2*match_range)]
    for i in range (0,match_range):
        if i == 0:
            NAME[match_range-1] ='HOMO'
            NAME[-match_range] ='LUMO'
        else:
            NAME[match_range-1-i]='HOMO-'+str(i)
            NAME[-match_range+i]='LUMO+'+str(i)
    HEADER = ['x-dis','y-dis','z-dis','gamma']+NAME
    for i in range (1,11):
        HEADER = HEADER + ['EXC{}'.format(i)]
    for i in range (1,11):
        HEADER = HEADER + ['PROB{}'.format(i)]
    ##File_write
    saved_file = os.path.join(NWchem,'processed_csv','{}.csv'.format(Folder))
    with open (saved_file,'w', newline='') as csvfile:
        writer = csv.writer (csvfile)
        writer.writerow (HEADER)
        for i in range(len(xdis)):
            writer.writerow ([xdis[i],ydis[i],zdis[i],theta[i]]+
                              [MO[j][i] for j in range(match_range*2)]+
                              [excited_states[j][i] for j in range(exc_num)]+
                              [osc_strength[j][i] for j in range(exc_num)])
    sortfile = pandas.read_csv(saved_file)
    newfile = sortfile.sort_values (['x-dis','y-dis','z-dis','gamma']).to_csv(index=False)
    with open (saved_file,'w',newline='') as nsfile:
        nsfile.write(newfile)
    return (MO)



#rotation_scan ('pentacene.xyz','pentacene_x0-0_y0.2-4_z3.5_theta0-90',0,0,0,4,3.5,3.5,0.2,0,90,5,'nw')
#shifting_scan ('pentacene.xyz', 'core testing 2',0,1,0,4,3.5,3.5,0.2,'xyz')
#


# =============================================================================
# NWchem = os.path.abspath(os.curdir)
# di = os.path.join(NWchem,'calculation_data','weird bug test') #Folder directory  
# #create the Folder path
# if os.path.exists(di):
#     shutil.rmtree(di)
# os.mkdir(di)
# #generate molecule coordinate 
# originalFilePath = os.path.join (NWchem,'original_xyz_file','pentacene.xyz')
# molecule0 = starting_molecule(originalFilePath).get_coordinate()
# molecule1 = starting_molecule(originalFilePath).rotate(88)
# #print(molecule1)
# file_ij = os.path.join(di,'weird.-.')
# write_file_xyz(file_ij,molecule0,molecule1,0,0,10)
# =============================================================================
















