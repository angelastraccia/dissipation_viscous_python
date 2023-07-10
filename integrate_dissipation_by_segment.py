# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:25:21 2022

@author: GALADRIEL_GUEST
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:51:35 2022

@author: GALADRIEL_GUEST
"""
import pyvista as pv
import numpy as np
from pyvista import examples
import pickle
import time
import os
import glob
import time
import scipy.io
import scipy
from tqdm import tqdm


#%%

def load_dict(name):
    """


    Parameters
    ----------
    name : str. path + name of the dictionary one wants to load

    Returns
    -------
    b : the loaded dictionary

    """
    with open(name + ".pkl", "rb") as handle:
        b = pickle.load(handle)
    return b

def save_dict(dico, name):
    """
    Parameters
    ----------
    dico : dictionary one wants to save
    name : str. path + name of the dictionary

    Returns
    -------
    None.

    """

    with open(name + ".pkl", "wb") as f:
        pickle.dump(dico, f)

def get_list_files_dat(pinfo, case, num_cycle):
    """


    Parameters
    ----------
    pinfo : str, patient information, composed of pt/vsp + number.
    num_cycle : int, number of the cycle computed
    case : str, baseline or vasospasm

    Returns
    -------
    onlyfiles : list of .dat files for the current patient, for the case and cycle wanted.
    """

    num_cycle = str(num_cycle)

    pathwd = "L:/vasospasm/" + pinfo + "/" + case + "/3-computational/hyak_submit/"

    os.chdir(pathwd)
    onlyfiles = []
    for file in glob.glob("*.dat"):
        if pinfo + "_" + case + "_cycle" + num_cycle in file:
            onlyfiles.append(file)
    indices = [l[:-4] for l in onlyfiles]

    return onlyfiles, indices,pathwd


def integrate_data_in_region(mesh_data):

    # Integrate values for entire volume
    integrated_volume = mesh_data.integrate_data()
    center = integrated_volume.points[0]
    volume = integrated_volume['Volume'][0]
    integrated_epsilon = integrated_volume['epsilon'][0]
    epsilon_normalized_by_volume = integrated_epsilon/volume
    # print(volume)
    # print(integrated_epsilon)
    
    return integrated_epsilon, volume, epsilon_normalized_by_volume
    

#%%

# # Define patient info
# pinfo = input('Patient -- ')
# case = input('Condition -- ')


def main(pinfo,case):
    num_cycle = 2
    
    # Paths
    geometry_path = "L:/vasospasm/"+pinfo+"/" + case + "/1-geometry/"
    results_path = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/" + "dissipation_viscous/"
    
    # Read in STL of surface
    fname_stl_m = geometry_path + pinfo + '_' + case + '_final.stl'
    stl_surf_m = pv.read(fname_stl_m)
    
    # Read in indices corresponding to each region in the original mesh
    dcenterindices = load_dict(results_path +'centerpoint_indices_' + pinfo + '_' + case)
    
    # Get list of vessel names
    all_region_names = [dcenterindices.get("indices{}".format(i_region))[0] for i_region in range(0,len(dcenterindices))]
    num_regions = len(all_region_names)
    
    # Instantiate dictionaries
    ddissipation = {}
    
    #%% LOOP THROUGH DATA
    data_first = 0
    data_last = 30
    data_skip = 1
    
    # data_indices = range(0,30,3)
    tic_total = time.perf_counter()
    for i_data in tqdm(range(data_first,data_last,data_skip)):
    
        ddissipation_tstep = {}
    
        # Get list of filenames
        onlyfiles, data_indices, pathwd = get_list_files_dat(pinfo,case,num_cycle)
        data_filename = results_path + data_indices[i_data] + '_epsilon.dat'
        
        # Read in Tecplot data file
        tic = time.perf_counter()
        reader = pv.get_reader(data_filename)
        mesh_data_imported = reader.read()
        mesh_data_final = mesh_data_imported[0]
        print(mesh_data_final.cell_data)
        toc = time.perf_counter()
        time_minutes = (toc-tic)/60
        print(f"Data import took {time_minutes:0.4f} minutes")
        
        for i_region in range(0,num_regions):
            region_name = dcenterindices.get("indices{}".format(i_region))[0]
            centerpoint_indices = dcenterindices.get("indices{}".format(i_region))[1]
            
            region_mesh = mesh_data_final.extract_cells(centerpoint_indices)
            
            integrated_epsilon, volume, epsilon_normalized_by_volume = integrate_data_in_region(region_mesh)
            ddissipation_tstep["dissipation{}".format(i_region)] = region_name, integrated_epsilon, volume, epsilon_normalized_by_volume
        
        ddissipation["{}".format(data_indices[i_data])] = ddissipation_tstep
    
    toc_total = time.perf_counter()
    time_hours = (toc_total-tic_total)/60
    print(f"All integrations  took {time_hours:3.1f} minutes")
    
    #%% Save dictionaries
    
    save_dict(ddissipation, results_path +'dissipation_' + pinfo + '_' + case)
    print('Saved dissipation dictionary')

main('pt40','vasospasm')
main('pt39','baseline')











