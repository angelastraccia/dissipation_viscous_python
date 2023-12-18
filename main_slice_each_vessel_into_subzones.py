# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:46:17 2023

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
import math
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

def clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,mesh_data,stl_surf_m,plot_flag):
    vessel_name = dpoints.get("points{}".format(i_vessel))[0]
    vessel_points = dpoints.get("points{}".format(i_vessel))[1] 
    vessel_vectors = dvectors.get("vectors{}".format(i_vessel))[1]
    vessel_radii = dradii.get("radii{}".format(i_vessel))[1]

    plane_center = (vessel_points[slice_index,0],vessel_points[slice_index,1],vessel_points[slice_index,2])
    plane_normal = (vessel_vectors[slice_index,0],vessel_vectors[slice_index,1],vessel_vectors[slice_index,2])

    # Slice through the data
    slice_plane = pv.Plane(center = plane_center, direction = plane_normal, i_size = vessel_radii[0]*3, j_size = vessel_radii[0]*3)
    mesh_clipped = mesh_data.clip_surface(slice_plane, invert=invert_status)
    
    
    # Determine connectivity
    mesh_clipped_connected = mesh_clipped.connectivity()


    # # Plot clipped surface
    # p = pv.Plotter()
    # p.add_mesh(mesh_clipped_connected, scalars='RegionId',label=vessel_name)
    # p.add_legend(size=(.2, .2), loc='upper right')
    # p.add_mesh(stl_surf_m, opacity=0.3)
    # p.show()

    #which_branch_id = int(input('Which branch is correct -- '))
    
    regionID_list = np.unique(mesh_clipped_connected['RegionId'])

    # cycle through each branch
    distance_list = []

    for regionID in regionID_list:
        #print(regionID)

        # Extract the region
        regionID_cells = np.where(
            mesh_clipped_connected['RegionId'] == regionID)[0]

        regionID_volume = mesh_clipped_connected.extract_cells(regionID_cells)
        integrated_volume_regionID = regionID_volume.integrate_data()

        # Determine the center of the region
        volume_center = integrated_volume_regionID.points[0]
        
        # Determine the midpoint of the centerline
        centerline_midpoint_index = int(np.round(np.shape(vessel_points)[0] /2))
        centerline_midpoint = vessel_points[centerline_midpoint_index,:]
        
        # Calculate the distance between the center of the volume and the centerpoint of the centerline
        distance_between_center_midpoint = math.dist(volume_center,centerline_midpoint)
        distance_list.append(distance_between_center_midpoint)
        #print(distance_between_center_midpoint)

        if plot_flag == 1:
            p = pv.Plotter()
            p.add_mesh(stl_surf_m, opacity=0.3)
            p.add_mesh(regionID_volume,label=vessel_name)
            p.add_legend(size=(.2, .2), loc='upper right')
            p.show()
        
    # # Determine the region id by minimizing

    which_branch_id = np.argmin(distance_list) #int(input('region id -- ')) #
    
    # Identify which cells are associated with the branch
    clipped_cells = np.where(mesh_clipped_connected['RegionId']==which_branch_id)[0]
    clipped_volume = mesh_clipped_connected.extract_cells(clipped_cells)

    # Plot segment
    if plot_flag == 1:
        p = pv.Plotter()
        p.add_mesh(stl_surf_m, opacity=0.3)
        p.add_mesh(clipped_volume,label=vessel_name)
        p.add_legend(size=(.2, .2), loc='upper right')
        p.show()
    
    return clipped_volume, which_branch_id

def clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,mesh_data,stl_surf_m, plot_flag):
    vessel_points = dpoints.get("points{}".format(i_vessel))[1]
    vessel_vectors = dvectors.get("vectors{}".format(i_vessel))[1]
    vessel_radii = dradii.get("radii{}".format(i_vessel))[1]
    
    plane_center = (vessel_points[slice_index,0],vessel_points[slice_index,1],vessel_points[slice_index,2])
    plane_normal = (vessel_vectors[slice_index,0],vessel_vectors[slice_index,1],vessel_vectors[slice_index,2])
        
    # Slice through the data
    slice_plane = pv.Plane(center = plane_center, direction = plane_normal, i_size = vessel_radii[0]*3, j_size = vessel_radii[0]*3)
    mesh_clipped = mesh_data.clip_surface(slice_plane, invert=invert_status)
    
    # Determine connectivity
    mesh_clipped_connected = mesh_clipped.connectivity()

    # Extract largest segment
    largest_region = mesh_clipped_connected.extract_largest()
    
    # Plot region
    if plot_flag == 1:
        p = pv.Plotter()
        p.add_mesh(stl_surf_m, opacity=0.3)
        p.add_mesh(largest_region, scalars='RegionId')
        p.show()
    
    return largest_region

#%%

# Define patient info
pinfo = 'pt40' #input('Patient -- ')
case = 'vasospasm' #baseline' #input('Condition -- ')
plot_flag = 1 #int(input('Plot figures? 1 = yes -- '))
num_cycle = 2

# Paths
geometry_path = "L:/vasospasm/"+pinfo+"/" + case + "/1-geometry/"
dissipation_path = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/" + "dissipation_viscous/"
results_path = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/" + "dissipation_viscous/"

# Read in STL of surface
fname_stl_m = geometry_path + pinfo + '_' + case + '_final.stl'
stl_surf_m = pv.read(fname_stl_m)

# Read in centerlines, vectors, etc.
centerline_data_path = "L:/vasospasm/"+pinfo+"/" + case + "/4-results/pressure_resistance/"
dvectors = load_dict(centerline_data_path +"vectors_" + pinfo + "_" + case)
dpoints = load_dict(centerline_data_path +"points_" + pinfo + "_" + case)
dradii= load_dict(centerline_data_path +"radii_" + pinfo + "_" + case)
dcenterindices = load_dict(dissipation_path +'centerpoint_indices_' + pinfo + '_' + case)

# Get list of filenames
i_data = 0
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

#%% Identify centerpoints and normal vectors

# Plotting settings
plot_flag = 0
cmap = plt.get_cmap('inferno')
num_vessels = len(dpoints)
dcenterindices_vessel_subzones = {}

for i_vessel in range(0,num_vessels):
    vessel_name = dpoints["points{}".format(i_vessel)][0]
    centerpoints = dpoints["points{}".format(i_vessel)][1] 
    num_points = len(centerpoints)
    vectors = dvectors["vectors{}".format(i_vessel)][1] 

    # Define a discretized color map
    colors = cmap(np.linspace(0, .8, num_points+1))
    
    #%Extract vessel zone
    region_name = dcenterindices.get("indices{}".format(i_vessel))[0]
    centerpoint_indices = dcenterindices.get("indices{}".format(i_vessel))[1]
    
    region_mesh = mesh_data_final.extract_cells(centerpoint_indices)
    
    #% Slice into smaller zones
    num_slices = num_points - 1
    tuple_centerindices = (vessel_name,)
    
    p = pv.Plotter()
    p.add_mesh(stl_surf_m , opacity=0.2, color='white')
    
    plot_original_points = pv.PolyData(centerpoints)
    p.add_mesh(plot_original_points,label=vessel_name,color='b')
    plot_original_points["vectors"] = vectors*0.001
    plot_original_points.set_active_vectors("vectors")
    p.add_mesh(plot_original_points.arrows, lighting=False)
    
    
    # Slice with first plane and centerpoint
    # Choose direction of slice
    vessel_subzone_first_slice_index = 1
    invert_status = False
        
    vessel_subzone = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,vessel_subzone_first_slice_index,invert_status,region_mesh,stl_surf_m, plot_flag)
    
    # Identify the centerpoints for the segment with respect to the original mesh
    vessel_subzone_centers = vessel_subzone.cell_centers()
    vessel_subzone_center_points = vessel_subzone_centers.points
    vessel_subzone_center_point_indices = mesh_data_final.find_containing_cell(vessel_subzone_center_points)
    vessel_subzone_segment_extracted = mesh_data_final.extract_cells(vessel_subzone_center_point_indices)
    
    p.add_mesh(vessel_subzone_segment_extracted,color=colors[vessel_subzone_first_slice_index])
    
    tuple_centerindices = tuple_centerindices + (vessel_subzone_center_point_indices,)
    
    #% Slice intermediate locations with two planes
     
    for vessel_subzone_first_slice_index in range(2,num_slices+1):
        invert_status = False
        vessel_subzone_largest = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,vessel_subzone_first_slice_index,invert_status,region_mesh,stl_surf_m, plot_flag)
        
        if invert_status == False:
            invert_status = True
        else:
            invert_status = False
        
        vessel_subzone_second_slice_index = vessel_subzone_first_slice_index - 1 
        vessel_subzone, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,vessel_subzone_second_slice_index,invert_status,vessel_subzone_largest,stl_surf_m,plot_flag)
        
        # Identify the centerpoints for the segment with respect to the original mesh
        vessel_subzone_centers = vessel_subzone.cell_centers()
        vessel_subzone_center_points = vessel_subzone_centers.points
        vessel_subzone_center_point_indices = mesh_data_final.find_containing_cell(vessel_subzone_center_points)
        vessel_subzone_segment_extracted = mesh_data_final.extract_cells(vessel_subzone_center_point_indices)
        
        
        p.add_mesh(vessel_subzone_segment_extracted,color=colors[vessel_subzone_first_slice_index])
            
        tuple_centerindices = tuple_centerindices + (vessel_subzone_center_point_indices,)
    
    p.add_mesh(region_mesh,opacity=0.7)
    p.show()
       
    # Save centerindices
    dcenterindices_vessel_subzones["indices{}".format(i_vessel)] = tuple_centerindices

#%%

save_dict(dcenterindices_vessel_subzones, results_path +'centerpoint_indices_vessel_subzones_' + pinfo + '_' + case)

print('Saved dictionaries')









