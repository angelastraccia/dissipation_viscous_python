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

pinfo = input('Patient number -- ') 
case = input ('Condition -- ')

results_path = 'L:/vasospasm/' + pinfo + '/' + case + '/4-results/pressure_resistance/'        plot_spaced_points["vectors"] = downsampled_normal*0.001

fname_stl_m = 'L:/vasospasm/' + pinfo + '/' + case + \
    '/1-geometry/' + pinfo + '_' + case + '_final.stl'
stl_surf_m = pv.read(fname_stl_m)

dpoints_original = load_dict(results_path +"points_original_" + pinfo + "_" + case)
dvectors_original = load_dict(results_path +"vectors_original_" + pinfo + "_" + case)


dpoints_cleaned = load_dict(results_path + 'points_' + pinfo + '_' + case)
dvectors_cleaned = load_dict(results_path + 'vectors_' + pinfo + '_' + case)

#%%
for i_vessel in range(len(dpoints_original)):

    vessel_name = dpoints_original["points{}".format(i_vessel)][0]
    # Plot STL and points
    check_vessels_plot = pv.Plotter()
    check_vessels_plot.add_mesh(stl_surf_m, opacity=0.3)

    plot_original_points = pv.PolyData(
        dpoints_original["points{}".format(i_vessel)][1])
    check_vessels_plot.add_mesh(plot_original_points,label=vessel_name,color='w')
    plot_original_points["vectors"] = dvectors_original["vectors{}".format(i_vessel)][1]*0.001
    
    plot_original_points.set_active_vectors("vectors")
    check_vessels_plot.add_mesh(plot_original_points.arrows, lighting=False)
    
    plot_cleaned_points = pv.PolyData(
        dpoints_cleaned["points{}".format(i_vessel)][1])
    check_vessels_plot.add_mesh(plot_cleaned_points,label=vessel_name,color='k')
    
    check_vessels_plot.add_legend(size=(.2, .2), loc='upper right')
    check_vessels_plot.show()
    
#%% Plot extracted regions

results_path = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/" + "dissipation_viscous/"

# Read in indices corresponding to each region in the original mesh
dcenterindices = load_dict(results_path +'centerpoint_indices_' + pinfo + '_' + case)

# Get list of vessel names
all_region_names = [dcenterindices.get("indices{}".format(i_region))[0] for i_region in range(0,len(dcenterindices))]
num_regions = len(all_region_names)




