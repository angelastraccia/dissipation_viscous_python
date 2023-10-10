# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:51:51 2023

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
import matplotlib.pyplot as plt



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



#%% Plot original points

# pinfo = input('Patient number -- ') 
# case = input ('Condition -- ')

# patient_list = ['pt2', 'pt7','pt8','pt10','pt12','pt21','pt28',
#                 'pt29','pt30','pt33','pt36','pt39','pt40',
#                 'vsp4','vsp5','vsp6','vsp7','vsp16',
#                 'vsp17','vsp19','vsp23','vsp24','vsp25','vsp26']

# condition_list = ["baseline","vasospasm"]

patient_list = ['pt40']
condition_list= ['baseline','vasospasm']

for pinfo in patient_list:
    for case in condition_list:
        
        print(pinfo + " " + case)

        pressure_path = 'L:/vasospasm/' + pinfo + '/' + case + '/4-results/pressure_resistance/'     
        dissipation_path = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/" + "dissipation_viscous/"   
        
        fname_stl_m = 'L:/vasospasm/' + pinfo + '/' + case + \
            '/1-geometry/' + pinfo + '_' + case + '_final.stl'
        stl_surf_m = pv.read(fname_stl_m)
        
        dpoints_original = load_dict(pressure_path +"points_original_" + pinfo + "_" + case)
        dvectors_original = load_dict(pressure_path +"vectors_original_" + pinfo + "_" + case)
        
        
        dpoints_cleaned = load_dict(pressure_path + 'points_' + pinfo + '_' + case)
        dvectors_cleaned = load_dict(pressure_path + 'vectors_' + pinfo + '_' + case)
        
        
        # Get list of filenames
        i_data = 0
        num_cycle = 2
        onlyfiles, data_indices, pathwd = get_list_files_dat(pinfo,case,num_cycle)
        data_filename = dissipation_path + data_indices[i_data] + '_epsilon.dat'
        
        # Read in Tecplot data file
        tic = time.perf_counter()
        reader = pv.get_reader(data_filename)
        mesh_data_imported = reader.read()
        mesh_data_final = mesh_data_imported[0]
        print(mesh_data_final.cell_data)
        toc = time.perf_counter()
        time_minutes = (toc-tic)/60
        print(f"Data import took {time_minutes:0.4f} minutes")
        
        
        # Read in indices corresponding to each region in the original mesh
        dcenterindices = load_dict(dissipation_path +'centerpoint_indices_' + pinfo + '_' + case)
        
        # #%%
        # for i_vessel in range(len(dpoints_original)):
        
        #     vessel_name = dpoints_original["points{}".format(i_vessel)][0]
        #     # Plot STL and points
        #     check_vessels_plot = pv.Plotter()
        #     check_vessels_plot.add_mesh(stl_surf_m, opacity=0.3)
        
        #     plot_original_points = pv.PolyData(
        #         dpoints_original["points{}".format(i_vessel)][1])
        #     check_vessels_plot.add_mesh(plot_original_points,label=vessel_name,color='w')
        #     plot_original_points["vectors"] = dvectors_original["vectors{}".format(i_vessel)][1]*0.001
            
        #     plot_original_points.set_active_vectors("vectors")
        #     check_vessels_plot.add_mesh(plot_original_points.arrows, lighting=False)
            
        #     plot_cleaned_points = pv.PolyData(
        #         dpoints_cleaned["points{}".format(i_vessel)][1])
        #     check_vessels_plot.add_mesh(plot_cleaned_points,label=vessel_name,color='k')
            
        #     check_vessels_plot.add_legend(size=(.2, .2), loc='upper right')
        #     check_vessels_plot.show()
            
        #% Plot extracted regions
        
        # Get list of vessel names
        all_vessel_names = [dcenterindices.get("indices{}".format(i_region))[0] for i_region in range(0,len(dcenterindices))]
        num_vessels = len(all_vessel_names)
        vessel_name_list = ['L_MCA','R_MCA','L_A1','L_A2','R_A1','R_A2',
                            'L_PCOM','L_P1','L_P2','R_PCOM','R_P1','R_P2',
                            'BAS','L_TICA','R_TICA']
        
        #Define a discretized color map
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, num_vessels))
        
        plot_segments = pv.Plotter()
        
        plot_segments.add_mesh(stl_surf_m, opacity=0.3)
        plot_segments.background_color = 'w'
        
        
        # Cycle through each branch
        for i_vessel in range(num_vessels):
            
            vessel_name = dcenterindices.get("indices{}".format(i_vessel))[0]
            #print(vessel_name)
            
            if vessel_name in vessel_name_list:
                branch_center_point_indices = dcenterindices.get("indices{}".format(i_vessel))[1]
            
                branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)
                
                # Plot the branch colored by the branch ID
                plot_segments.add_mesh(branch_segment_extracted, color=colors[i_vessel], label=vessel_name)
        
        plot_segments.add_legend(size=(.3, .3), loc='upper right',bcolor='w')
        plot_segments.show()
        



