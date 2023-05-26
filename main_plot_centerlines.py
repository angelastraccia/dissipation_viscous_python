# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:51:51 2023

@author: GALADRIEL_GUEST
"""
import pickle
import pyvista as pv


def load_dict(name):
    """

vsp
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



#%% Plot original points

pinfo = input('Patient number -- ') 
case = input ('Condition -- ')

results_path = 'L:/vasospasm/' + pinfo + '/' + case + '/4-results/pressure_resistance/'

fname_stl_m = 'L:/vasospasm/' + pinfo + '/' + case + \
    '/1-geometry/' + pinfo + '_' + case + '_final.stl'
stl_surf_m = pv.read(fname_stl_m)

dpoints_original = load_dict(results_path +"points_original_" + pinfo + "_" + case)
dvectors_original = load_dict(results_path +"vectors_original_" + pinfo + "_" + case)


dpoints_cleaned = load_dict(results_path + 'points_' + pinfo + '_' + case)
dvectors_cleaned = load_dict(results_path + 'vectors_' + pinfo + '_' + case)

for i_vessel in range(len(dpoints_original)):

    vessel_name = dpoints_original["points{}".format(i_vessel)][0]
    # Plot STL and points
    check_vessels_plot = pv.Plotter()
    check_vessels_plot.add_mesh(stl_surf_m, opacity=0.3)

    plot_original_points = pv.PolyData(
        dpoints_original["points{}".format(i_vessel)][1])
    check_vessels_plot.add_mesh(plot_original_points,label=vessel_name,color='w')
    
    plot_cleaned_points = pv.PolyData(
        dpoints_cleaned["points{}".format(i_vessel)][1])
    check_vessels_plot.add_mesh(plot_cleaned_points,label=vessel_name,color='k')
    
    check_vessels_plot.add_legend(size=(.2, .2), loc='upper right')
    check_vessels_plot.show()
    
    
    




