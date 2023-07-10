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

        p = pv.Plotter()
        p.add_mesh(stl_surf_m, opacity=0.3)
        p.add_mesh(regionID_volume,label=vessel_name)
        p.add_legend(size=(.2, .2), loc='upper right')
        p.show()
        
    # # Determine the region id by minimizing

    which_branch_id = int(input('region id -- ')) #np.argmin(distance_list) #
    
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

def clip_by_dat_file(dat_filename,mesh_data,invert_status,stl_surf_m,plot_flag):
    
    geom_data = np.genfromtxt(dat_filename,
                         skip_header=1)
    plane_center = (geom_data[0],geom_data[1],geom_data[2])
    plane_normal = (geom_data[3],geom_data[4],geom_data[5])
    slice_plane = pv.Plane(center = plane_center, direction = plane_normal, i_size = 0.1, j_size = 0.1)

    # Clip data
    mesh_clipped = mesh_data.clip_surface(slice_plane, invert=invert_status)
    mesh_clipped_connected = mesh_clipped.connectivity()

    # Extract largest segment and plot it
    largest_region = mesh_clipped_connected.extract_largest()

    # Plot region
    if plot_flag == 1:
        p = pv.Plotter()
        p.add_mesh(stl_surf_m, opacity=0.3)
        p.add_mesh(largest_region, scalars='RegionId')
        p.show()

    return largest_region

def clip_mesh(dat_filename,mesh_data,invert_status,stl_surf_m,plot_flag):
    
    geom_data = np.genfromtxt(dat_filename,
                         skip_header=1)
    plane_center = (geom_data[0],geom_data[1],geom_data[2])
    plane_normal = (geom_data[3],geom_data[4],geom_data[5])
    slice_plane = pv.Plane(center = plane_center, direction = plane_normal, i_size = 0.1, j_size = 0.1)

    # Clip data
    mesh_clipped = mesh_data.clip_surface(slice_plane, invert=invert_status)

    # Plot region
    if plot_flag == 1:
        p = pv.Plotter()
        p.add_mesh(stl_surf_m, opacity=0.3)
        p.add_mesh(mesh_clipped)
        p.show()
    
    return mesh_clipped

#%%

# Define patient info
pinfo = input('Patient -- ')
case = input('Condition -- ')
plot_flag = int(input('Plot figures? 1 = yes -- '))
num_cycle = 2


# Identifies the directory with the case_info.mat file
dinfo = scipy.io.loadmat(
    "L:/vasospasm/" + pinfo + "/" + case + "/3-computational/case_info.mat"
)

# Extract the variation input number
variation_input_case_info = dinfo.get("variation_input")
variation_input = variation_input_case_info[0][0]

# Paths
geometry_path = "L:/vasospasm/"+pinfo+"/" + case + "/1-geometry/"
results_path = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/" + "dissipation_viscous/"

# Read in STL of surface
fname_stl_m = geometry_path + pinfo + '_' + case + '_final.stl'
stl_surf_m = pv.read(fname_stl_m)

# Read in centerlines, vectors, etc.
centerline_data_path = "L:/vasospasm/"+pinfo+"/" + case + "/4-results/pressure_resistance/"
dvectors = load_dict(centerline_data_path +"vectors_" + pinfo + "_" + case)
dpoints = load_dict(centerline_data_path +"points_" + pinfo + "_" + case)
dradii= load_dict(centerline_data_path +"radii_" + pinfo + "_" + case)

#% Instantiate dictionaries
dsliceindex, dbranchid,dcenterindices = {}, {},{}

# Get list of filenames
i_data = 0
onlyfiles, data_indices, pathwd = get_list_files_dat(pinfo,case,num_cycle)
data_filename = results_path + data_indices[i_data] + '_epsilon.dat'

# data_filename = results_path + pinfo + '_' + case + '_epsilon.dat'

# Get list of vessel names
all_vessel_names = [dpoints.get("points{}".format(i_vessel))[0] for i_vessel in range(0,len(dpoints))]
num_vessels = len(all_vessel_names)

# Read in Tecplot data file
tic = time.perf_counter()
reader = pv.get_reader(data_filename)
mesh_data_imported = reader.read()
mesh_data_final = mesh_data_imported[0]
print(mesh_data_final.cell_data)
toc = time.perf_counter()
time_minutes = (toc-tic)/60
print(f"Data import took {time_minutes:0.4f} minutes")


#%% Isolate the CoW

# Define basilar 
vessel_name = 'BAS'
BAS_vessel_index = all_vessel_names.index(vessel_name)

vessel_points = dpoints.get("points{}".format(BAS_vessel_index))[1]
vessel_vectors = dvectors.get("vectors{}".format(BAS_vessel_index))[1]

# Find maximum value in z for end points
BAS_first_point_z = vessel_points[0,2]
BAS_last_point_z = vessel_points[-1,2]

# Select point with the highest z coordinate for the basilar slice
if BAS_first_point_z > BAS_last_point_z:
    BAS_slice_index = 0
else:
    BAS_slice_index= -1
    
# Determine which is the point associated with the first cut plane (beginning or end of points)

if BAS_slice_index == 0:
    invert_status = True
else:
    invert_status = False

# Plot the points along the basilar artery
if plot_flag == 1:
    plot_vessel_points_vectors = pv.Plotter()
    plot_points = pv.PolyData(vessel_points)
    plot_vessel_points_vectors.add_mesh(stl_surf_m, opacity=0.3)
    plot_vessel_points_vectors.add_mesh(plot_points)
    plot_points["vectors"] = vessel_vectors*0.001
    plot_points.set_active_vectors("vectors")
    plot_vessel_points_vectors.add_mesh(
        plot_points.arrows, lighting=False)
    plot_vessel_points_vectors.show()
    
    mid_slice_index = int(input('Which point defines the CoW slice? -- '))

# Switch the direction of the slicing compared to BAS slice
else:
    mid_slice_index = 6


if BAS_slice_index == 0:
    invert_status = False
else:
    invert_status = True

tic = time.perf_counter()
cow_only = clip_mesh_by_largest(dpoints,dvectors,dradii,BAS_vessel_index,mid_slice_index,invert_status,mesh_data_final,stl_surf_m,plot_flag)
toc = time.perf_counter()
time_minutes = (toc-tic)/60

cow_centers = cow_only.cell_centers()
cow_center_points = cow_centers.points
cow_center_point_indices = mesh_data_final.find_containing_cell(cow_center_points)
cow_segment_extracted = mesh_data_final.extract_cells(cow_center_point_indices)

# plot = pv.Plotter()
# plot.add_mesh(cow_segment_extracted)
# plot.add_mesh(stl_surf_m,opacity=0.3)
# plot.show()

# Store slice index and invert status in dictionary
i_store = num_vessels
dsliceindex["sliceindex{}".format(i_store)] = "cow", mid_slice_index, invert_status
dcenterindices["indices{}".format(i_store)] = 'cow', cow_center_point_indices

print(f"Isolated CoW took {time_minutes:0.4f} minutes")

#%% Split into anterior and posterior

tic = time.perf_counter()

anterior_posterior_geom_data_filename = geometry_path + 'anterior_posterior.dat'
anterior_geom_data_filename = geometry_path + 'anterior.dat'
posterior_geom_data_filename = geometry_path + 'posterior.dat'

# If the CoW can be split between anterior and posterior
if os.path.isfile(anterior_posterior_geom_data_filename) == True:

    # Anterior
    invert_status = False
    
    # If the variation is missing a segment in the anterior circulation, keep all regions
    if variation_input == 2 or variation_input == 5 or variation_input == 6:
        anterior = clip_mesh(anterior_posterior_geom_data_filename,cow_only,invert_status,stl_surf_m,plot_flag)
        
    # Otherwise keep the largest region only
    else:
        anterior = clip_by_dat_file(anterior_posterior_geom_data_filename,cow_only,invert_status,stl_surf_m,plot_flag)
    
    anterior_centers = anterior.cell_centers()
    anterior_center_points = anterior_centers.points
    anterior_center_point_indices = mesh_data_final.find_containing_cell(anterior_center_points)
    anterior_segment_extracted = mesh_data_final.extract_cells(anterior_center_point_indices)
    
    i_store = num_vessels+1   
    dcenterindices["indices{}".format(i_store)] = 'anterior', anterior_center_point_indices    
    
    # Posterior
    invert_status = True
    
    # If the variation is missing a segment in the posterior circulation, keep all regions
    if variation_input == 3 or variation_input == 4:
        posterior = clip_mesh(anterior_posterior_geom_data_filename,cow_only,invert_status,stl_surf_m,plot_flag)
    # Otherwise keep the largest region only
    else:
        posterior = clip_by_dat_file(anterior_posterior_geom_data_filename,cow_only,invert_status,stl_surf_m,plot_flag)
    
    posterior_centers = posterior.cell_centers()
    posterior_center_points = posterior_centers.points
    posterior_center_point_indices = mesh_data_final.find_containing_cell(posterior_center_points)
    posterior_segment_extracted = mesh_data_final.extract_cells(posterior_center_point_indices)
    
    i_store = num_vessels+2 
    dcenterindices["indices{}".format(i_store)] = 'posterior', posterior_center_point_indices
    
    toc = time.perf_counter()
    time_minutes = (toc-tic)/60
    print(f"Anterior/posterior slices  took {time_minutes:0.4f} minutes")

# Otherwise use separate anterior.dat and posterior.dat files
else:
    # Anterior
    invert_status = False
    
    # If the variation is missing a segment in the anterior circulation, keep all regions
    if variation_input == 2 or variation_input == 5 or variation_input == 6:
        anterior = clip_mesh(anterior_geom_data_filename,cow_only,invert_status,stl_surf_m,plot_flag)
        
    # Otherwise keep the largest region only
    else:
        anterior = clip_by_dat_file(anterior_geom_data_filename,cow_only,invert_status,stl_surf_m,plot_flag)
    
    anterior_centers = anterior.cell_centers()
    anterior_center_points = anterior_centers.points
    anterior_center_point_indices = mesh_data_final.find_containing_cell(anterior_center_points)
    anterior_segment_extracted = mesh_data_final.extract_cells(anterior_center_point_indices)
    
    i_store = num_vessels+1   
    dcenterindices["indices{}".format(i_store)] = 'anterior', anterior_center_point_indices    
    
    # Posterior
    invert_status = False
    
    # If the variation is missing a segment in the posterior circulation, keep all regions
    if variation_input == 3 or variation_input == 4:
        posterior = clip_mesh(posterior_geom_data_filename,cow_only,invert_status,stl_surf_m,plot_flag)
    # Otherwise keep the largest region only
    else:
        posterior = clip_by_dat_file(posterior_geom_data_filename,cow_only,invert_status,stl_surf_m,plot_flag)
    
    posterior_centers = posterior.cell_centers()
    posterior_center_points = posterior_centers.points
    posterior_center_point_indices = mesh_data_final.find_containing_cell(posterior_center_points)
    posterior_segment_extracted = mesh_data_final.extract_cells(posterior_center_point_indices)
    
    i_store = num_vessels+2 
    dcenterindices["indices{}".format(i_store)] = 'posterior', posterior_center_point_indices

toc = time.perf_counter()
time_minutes = (toc-tic)/60
print(f"Anterior/posterior slices  took {time_minutes:0.4f} minutes")

if plot_flag == 1:
    plot = pv.Plotter()
    plot.add_mesh(stl_surf_m, opacity=0.3)
    plot.add_mesh(anterior_segment_extracted,'r',opacity=0.3,label='anterior')
    plot.add_mesh(posterior_segment_extracted,'b',opacity=0.3,label='posterior')
    plot.background_color = 'w'
    plot.add_legend(size=(.1, .1), loc='upper right',bcolor='w')
    plot.show() 
    
    
#%% Slice vessels that require 1 slice
plot_flag = int(input('Plot figures? 1 = yes -- '))

#% Slice the basilar artery

vessel_name = 'BAS'
BAS_vessel_index = all_vessel_names.index(vessel_name)

vessel_points = dpoints.get("points{}".format(BAS_vessel_index))[1]
vessel_vectors = dvectors.get("vectors{}".format(BAS_vessel_index))[1]

# Find maximum value in z for end points
BAS_first_point_z = vessel_points[0,2]
BAS_last_point_z = vessel_points[-1,2]

# Select point with the highest z coordinate for the basilar slice
if BAS_first_point_z > BAS_last_point_z:
    BAS_slice_index = 0
else:
    BAS_slice_index= -1
    
# Determine which is the point associated with the first cut plane (beginning or end of points)
#BAS_slice_index = int(input('Which point defines the cut plane for the basilar artery? 0: first point, -1: last point -- '))
if BAS_slice_index == 0:
    invert_status = True
else:
    invert_status = False

tic = time.perf_counter()
BAS, BAS_branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,BAS_vessel_index,BAS_slice_index,invert_status,mesh_data_final,stl_surf_m,plot_flag)
toc = time.perf_counter()
time_minutes = (toc-tic)/60
print(f"BAS slice  took {time_minutes:0.4f} minutes")

# Identify the centerpoints for the segment with respect to the original mesh
BAS_centers = BAS.cell_centers()
BAS_center_points = BAS_centers.points
BAS_center_point_indices = mesh_data_final.find_containing_cell(BAS_center_points)
BAS_segment_extracted = mesh_data_final.extract_cells(BAS_center_point_indices)

# p = pv.Plotter()
# p.add_mesh(BAS_segment_extracted)
# p.add_mesh(stl_surf_m, opacity=0.3)
# p.show()

# Store slice index + invert status, branchid, and centerpoint indices in dictionaries
dsliceindex["sliceindex{}".format(BAS_vessel_index)] = vessel_name, BAS_slice_index, invert_status
dbranchid["branchid{}".format(BAS_vessel_index)] = vessel_name, BAS_branch_id
dcenterindices["indices{}".format(BAS_vessel_index)] = vessel_name, BAS_center_point_indices

#% One slice vessels

one_slice_vessels = ['L_MCA','R_MCA','L_A2','R_A2','L_P2','R_P2']  #
anterior_one_slice_vessels = ['L_MCA','R_MCA','L_A2','R_A2']
slice_index = 0
invert_status = True

for vessel_name in one_slice_vessels:
    i_vessel = all_vessel_names.index(vessel_name)
    
    tic = time.perf_counter()
    if vessel_name in anterior_one_slice_vessels:
        branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,anterior,stl_surf_m,plot_flag)
    else:
        branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,posterior,stl_surf_m,plot_flag)
    
    toc = time.perf_counter()
    time_minutes = (toc-tic)/60
    
    branch_centers = branch_mesh.cell_centers()
    branch_center_points = branch_centers.points
    branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
    branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

    # plot = pv.Plotter()
    # plot.add_mesh(branch_segment_extracted)
    # plot.add_mesh(stl_surf_m,opacity=0.3)
    # plot.show() 
    
    # Store slice index + invert status and branchid in dictionaries
    dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
    dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
    dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
    
    print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")

    
#%% Slice vessels that require 2 slices

two_slice_vessels = ['L_A1','R_A1','L_P1','R_P1'] #'
anterior_two_slice_vessels = ['L_A1','R_A1']

for vessel_name in two_slice_vessels:
    # Check that the collateral pathway is present
    if vessel_name in all_vessel_names:
        i_vessel = all_vessel_names.index(vessel_name)
        
        # Info for first slice
        slice_index = 0 
        invert_status = True 
        
        # Store slice index for first slice
        dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
        
        tic = time.perf_counter()
        if vessel_name in anterior_two_slice_vessels:
            first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,anterior,stl_surf_m,plot_flag)
            # Reverse direction of the slice
            if slice_index == 0:
                slice_index = -1
                invert_status = False
            else:
                slice_index = 0
                invert_status = True
            branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
        else:
            first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,posterior,stl_surf_m,plot_flag)
            # Reverse direction of the slice
            if slice_index == 0:
                slice_index = -1
                invert_status = False
            else:
                slice_index = 0
                invert_status = True

            branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
        
        toc = time.perf_counter()
        time_minutes = (toc-tic)/60
        
        branch_centers = branch_mesh.cell_centers()
        branch_center_points = branch_centers.points
        branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
        branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

        # plot = pv.Plotter()
        # plot.add_mesh(branch_segment_extracted)
        # plot.add_mesh(stl_surf_m,opacity=0.3)
        # plot.show() 

        # Store branchid in dictionary
        dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
        dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
        
        print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")
        
#% Pcoms that require 2 slices

Pcom_vessels = ['L_PCOM','R_PCOM']

for vessel_name in Pcom_vessels:
    # Check that the collateral pathway is present
    if vessel_name in all_vessel_names:
        i_vessel = all_vessel_names.index(vessel_name)

        # Info for first slice
        slice_index = -1 #0
        invert_status = False #True

        # Store slice index for first slice
        dsliceindex["sliceindex{}".format(
            i_vessel)] = vessel_name, slice_index, invert_status

        tic = time.perf_counter()

        first_clip = clip_mesh_by_largest(
              dpoints, dvectors, dradii, i_vessel, slice_index, invert_status, cow_only, stl_surf_m, plot_flag) #left,
        
        # Reverse direction of the slice
        if slice_index == 0:
            slice_index = -1
            invert_status = False
        else:
            slice_index = 0
            invert_status = True
            
        branch_mesh, branch_id = clip_mesh_by_regionid(
            dpoints, dvectors, dradii, i_vessel, slice_index, invert_status, first_clip, stl_surf_m,plot_flag)

        toc = time.perf_counter()
        time_minutes = (toc-tic)/60
        
        branch_centers = branch_mesh.cell_centers()
        branch_center_points = branch_centers.points
        branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
        branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

        # plot = pv.Plotter()
        # plot.add_mesh(branch_segment_extracted)
        # plot.add_mesh(stl_surf_m,opacity=0.3)
        # plot.show() 

        # Store branchid in dictionary
        dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
        dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
        
        print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")


#% Slice vessels that require 3 slices
three_slice_vessels = ['L_TICA','R_TICA'] 


# If missing a left Pcom, only slice with the .dat file on the right side
if variation_input == 7:
    
    vessel_name = 'R_TICA'
    i_vessel = all_vessel_names.index(vessel_name)
    
    slice_dat_filename = geometry_path + vessel_name.lower() + '.dat'
    
    tic = time.perf_counter()
    # Clip with .dat file
    invert_status = False
    first_clip = clip_by_dat_file(slice_dat_filename,anterior,invert_status,stl_surf_m,plot_flag)
   
    # Clip using last point
    slice_index = 0# -1
    invert_status = True #False
    
    
    second_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
    
    dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
    
    # Clip using first point
    slice_index = -1 #0 
    invert_status = False #True
    branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,second_clip,stl_surf_m,plot_flag)
    
    toc = time.perf_counter()
    time_minutes = (toc-tic)/60
    
    branch_centers = branch_mesh.cell_centers()
    branch_center_points = branch_centers.points
    branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
    branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

    # plot = pv.Plotter()
    # plot.add_mesh(branch_segment_extracted)
    # plot.add_mesh(stl_surf_m,opacity=0.3)
    # plot.show() 

    # Store branchid in dictionary
    dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
    dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
    
    toc = time.perf_counter()
    time_minutes = (toc-tic)/60
    
    print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")
    
    tic = time.perf_counter()
    
    #% Slice LTICA without .dat file
    vessel_name = 'L_TICA'
    i_vessel = all_vessel_names.index(vessel_name)
    
    # Clip using last point
    slice_index = 0 #-1
    invert_status = True # False
    first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,anterior,stl_surf_m,plot_flag)
    
    # Reverse direction of the slice
    if slice_index == 0:
        slice_index = -1
        invert_status = False
    else:
        slice_index = 0
        invert_status = True
    branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
    
    branch_centers = branch_mesh.cell_centers()
    branch_center_points = branch_centers.points
    branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
    branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)
    
    # Store branchid in dictionary
    dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
    dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
    
    toc = time.perf_counter()
    time_minutes = (toc-tic)/60
    
    print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")


# If missing a right Pcom, only slice with the .dat file on the left side
elif variation_input == 8:
    vessel_name = 'L_TICA'
    i_vessel = all_vessel_names.index(vessel_name)
    
    slice_dat_filename = geometry_path + vessel_name.lower() + '.dat'
    
    tic = time.perf_counter()
    # Clip with .dat file
    invert_status = False
    first_clip = clip_by_dat_file(slice_dat_filename,anterior,invert_status,stl_surf_m,plot_flag)
   
    # Clip using last point
    slice_index = -1
    invert_status = False
    second_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
    
    dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
    
    # Clip using first point
    # Reverse direction of the slice
    if slice_index == 0:
        slice_index = -1
        invert_status = False
    else:
        slice_index = 0
        invert_status = True
    branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,second_clip,stl_surf_m,plot_flag)
    
    toc = time.perf_counter()
    time_minutes = (toc-tic)/60
    
    branch_centers = branch_mesh.cell_centers()
    branch_center_points = branch_centers.points
    branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
    branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

    # plot = pv.Plotter()
    # plot.add_mesh(branch_segment_extracted)
    # plot.add_mesh(stl_surf_m,opacity=0.3)
    # plot.show() 

    # Store branchid in dictionary
    dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
    dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
    
    
    # Clip using last point
    slice_index = -1
    invert_status = False
    first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,anterior,stl_surf_m,plot_flag)
    
    dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
    
    tic = time.perf_counter()
    
    # Slice RTICA without .dat file
    vessel_name = 'R_TICA'
    i_vessel = all_vessel_names.index(vessel_name)
    
    # Clip using first point
    # Reverse direction of the slice
    if slice_index == 0:
        slice_index = -1
        invert_status = False
    else:
        slice_index = 0
        invert_status = True
    branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
    
    toc = time.perf_counter()
    time_minutes = (toc-tic)/60
    
    branch_centers = branch_mesh.cell_centers()
    branch_center_points = branch_centers.points
    branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
    branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)
    
    toc = time.perf_counter()
    time_minutes = (toc-tic)/60
    
    print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")
    
# Otherwise slice with .dat files for both sides    
else:
    for vessel_name in three_slice_vessels:
        tic = time.perf_counter()
        
        i_vessel = all_vessel_names.index(vessel_name)
        
        tic = time.perf_counter()
        slice_dat_filename = geometry_path + vessel_name.lower() + '.dat'
        
        # Clip with .dat file
        invert_status = False
        first_clip = clip_by_dat_file(slice_dat_filename,anterior,invert_status,stl_surf_m,plot_flag)
    
        toc = time.perf_counter()
        time_minutes = (toc-tic)/60

    
        # Clip using a point
        slice_index = 0
        invert_status = True
        second_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
        
        dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
        
        # Reverse direction of the slice
        if slice_index == 0:
            slice_index = -1
            invert_status = False
        else:
            slice_index = 0
            invert_status = True
        branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,second_clip,stl_surf_m,plot_flag)
        toc = time.perf_counter()
        time_minutes = (toc-tic)/60
    
        
        branch_centers = branch_mesh.cell_centers()
        branch_center_points = branch_centers.points
        branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
        branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)
    
        # plot = pv.Plotter()
        # plot.add_mesh(branch_segment_extracted)
        # plot.add_mesh(stl_surf_m,opacity=0.3)
        # plot.show() 
    
        # Store branchid in dictionary
        dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
        dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
        
        print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")
    



#%% Plot sliced segments

#Define a discretized color map
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1, num_vessels))

plot_segments = pv.Plotter()

plot_segments.add_mesh(stl_surf_m, opacity=0.3)
plot_segments.background_color = 'w'


# Cycle through each branch
for i_vessel in range(num_vessels):
    
    vessel_name = dcenterindices.get("indices{}".format(i_vessel))[0]
    print(vessel_name)
    branch_center_point_indices = dcenterindices.get("indices{}".format(i_vessel))[1]

    branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)
    
    # Plot the branch colored by the branch ID
    plot_segments.add_mesh(branch_segment_extracted, color=colors[i_vessel], label=vessel_name)

plot_segments.add_legend(size=(.3, .3), loc='upper right',bcolor='w')
plot_segments.show()


#%% Save dictionaries

save_dict(dbranchid, results_path +'branch_id_' + pinfo + '_' + case)
save_dict(dsliceindex, results_path +'slice_index_' + pinfo + '_' + case)
save_dict(dcenterindices, results_path +'centerpoint_indices_' + pinfo + '_' + case)

print('Saved dictionaries')


# #%% One slice vessels

# one_slice_vessels = ['L_A2']  #
# anterior_one_slice_vessels = ['L_MCA','R_MCA','L_A2','R_A2']
# slice_index = 0
# invert_status = True

# for vessel_name in one_slice_vessels:
#     i_vessel = all_vessel_names.index(vessel_name)
    
#     tic = time.perf_counter()
#     if vessel_name in anterior_one_slice_vessels:
#         branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,anterior,stl_surf_m,plot_flag)
#     else:
#         branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,posterior,stl_surf_m,plot_flag)
    
#     toc = time.perf_counter()
#     time_minutes = (toc-tic)/60
    
#     branch_centers = branch_mesh.cell_centers()
#     branch_center_points = branch_centers.points
#     branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
#     branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

#     # plot = pv.Plotter()
#     # plot.add_mesh(branch_segment_extracted)
#     # plot.add_mesh(stl_surf_m,opacity=0.3)
#     # plot.show() 
    
#     # Store slice index + invert status and branchid in dictionaries
#     dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
#     dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
#     dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
    
#     print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")


# #%% Slice vessels that require 2 slices

# two_slice_vessels = ['R_P1'] #
# anterior_two_slice_vessels = ['L_A1','R_A1']

# for vessel_name in two_slice_vessels:
#     # Check that the collateral pathway is present
#     if vessel_name in all_vessel_names:
#         i_vessel = all_vessel_names.index(vessel_name)
        
#         # Info for first slice
#         slice_index = -1 
#         invert_status = False
        
#         # Store slice index for first slice
#         dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
        
#         tic = time.perf_counter()
#         if vessel_name in anterior_two_slice_vessels:
#             first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,anterior,stl_surf_m,plot_flag)
#             # Reverse direction of the slice
#             if slice_index == 0:
#                 slice_index = -1
#                 invert_status = False
#             else:
#                 slice_index = 0
#                 invert_status = True
#             branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
#         else:
#             first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,posterior,stl_surf_m,plot_flag)
#             # Reverse direction of the slice
#             if slice_index == 0:
#                 slice_index = -1
#                 invert_status = False
#             else:
#                 slice_index = 0
#                 invert_status = True

#             branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
        
#         toc = time.perf_counter()
#         time_minutes = (toc-tic)/60
        
#         branch_centers = branch_mesh.cell_centers()
#         branch_center_points = branch_centers.points
#         branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
#         branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

#         # plot = pv.Plotter()
#         # plot.add_mesh(branch_segment_extracted)
#         # plot.add_mesh(stl_surf_m,opacity=0.3)
#         # plot.show() 

#         # Store branchid in dictionary
#         dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
#         dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
        
#         print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")












#%% pt39 with missing right A1


#%% Left
tic = time.perf_counter()
invert_status = False
left_geom_data_filename = geometry_path + 'left.dat'    
left = clip_by_dat_file(left_geom_data_filename,cow_only,invert_status,stl_surf_m,plot_flag)

left_centers = left.cell_centers()
left_center_points = left.points
left_center_point_indices = mesh_data_final.find_containing_cell(left_center_points)
left_segment_extracted = mesh_data_final.extract_cells(left_center_point_indices)

i_store = num_vessels+3
dcenterindices["indices{}".format(i_store)] = 'left', left_center_point_indices    

toc = time.perf_counter()
time_minutes = (toc-tic)/60
print(f"left slices  took {time_minutes:0.4f} minutes")

#%%
two_slice_vessels = ['L_A1'] #
anterior_two_slice_vessels = ['L_A1','R_A1']

for vessel_name in two_slice_vessels:
    # Check that the collateral pathway is present
    if vessel_name in all_vessel_names:
        i_vessel = all_vessel_names.index(vessel_name)
        
        # Info for first slice
        slice_index = 0 
        invert_status = True 
        
        # Store slice index for first slice
        dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
        
        tic = time.perf_counter()
        if vessel_name in anterior_two_slice_vessels:
            first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,left,stl_surf_m,plot_flag)
            # Reverse direction of the slice
            if slice_index == 0:
                slice_index = -1
                invert_status = False
            else:
                slice_index = 0
                invert_status = True
            branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
        else:
            first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,posterior,stl_surf_m,plot_flag)
            # Reverse direction of the slice
            if slice_index == 0:
                slice_index = -1
                invert_status = False
            else:
                slice_index = 0
                invert_status = True

            branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
        
        toc = time.perf_counter()
        time_minutes = (toc-tic)/60
        
        branch_centers = branch_mesh.cell_centers()
        branch_center_points = branch_centers.points
        branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
        branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

        # plot = pv.Plotter()
        # plot.add_mesh(branch_segment_extracted)
        # plot.add_mesh(stl_surf_m,opacity=0.3)
        # plot.show() 

        # Store branchid in dictionary
        dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
        dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
        
        print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")
        
#%% Right
tic = time.perf_counter()
invert_status = True
right_geom_data_filename = geometry_path + 'left.dat'    
right = clip_by_dat_file(left_geom_data_filename,cow_only,invert_status,stl_surf_m,plot_flag)

right_centers = right.cell_centers()
right_center_points = right.points
right_center_point_indices = mesh_data_final.find_containing_cell(right_center_points)
right_segment_extracted = mesh_data_final.extract_cells(right_center_point_indices)

i_store = num_vessels+4
dcenterindices["indices{}".format(i_store)] = 'right', right_center_point_indices    

toc = time.perf_counter()
time_minutes = (toc-tic)/60
print(f"left slices  took {time_minutes:0.4f} minutes")

#%%

three_slice_vessels = ['R_TICA']
for vessel_name in three_slice_vessels:
    tic = time.perf_counter()
    
    i_vessel = all_vessel_names.index(vessel_name)
    
    tic = time.perf_counter()
    slice_dat_filename = geometry_path + vessel_name.lower() + '.dat'
    
    # Clip with .dat file
    invert_status = False
    first_clip = clip_by_dat_file(slice_dat_filename,right,invert_status,stl_surf_m,plot_flag)

    toc = time.perf_counter()
    time_minutes = (toc-tic)/60


    # Clip using a point
    slice_index = 0
    invert_status = True
    second_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
    
    dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
    
    # Reverse direction of the slice
    if slice_index == 0:
        slice_index = -1
        invert_status = False
    else:
        slice_index = 0
        invert_status = True
    branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,second_clip,stl_surf_m,plot_flag)
    toc = time.perf_counter()
    time_minutes = (toc-tic)/60

    
    branch_centers = branch_mesh.cell_centers()
    branch_center_points = branch_centers.points
    branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
    branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

    # plot = pv.Plotter()
    # plot.add_mesh(branch_segment_extracted)
    # plot.add_mesh(stl_surf_m,opacity=0.3)
    # plot.show() 

    # Store branchid in dictionary
    dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
    dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
    
    print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")

#%%
Pcom_vessels = ['R_PCOM']

for vessel_name in Pcom_vessels:
    # Check that the collateral pathway is present
    if vessel_name in all_vessel_names:
        i_vessel = all_vessel_names.index(vessel_name)

        # Info for first slice
        slice_index = -1 #0
        invert_status = False #True

        # Store slice index for first slice
        dsliceindex["sliceindex{}".format(
            i_vessel)] = vessel_name, slice_index, invert_status

        tic = time.perf_counter()

        first_clip = clip_mesh_by_largest(
              dpoints, dvectors, dradii, i_vessel, slice_index, invert_status, right, stl_surf_m, plot_flag) #left,
        
        # Reverse direction of the slice
        if slice_index == 0:
            slice_index = -1
            invert_status = False
        else:
            slice_index = 0
            invert_status = True
            
        branch_mesh, branch_id = clip_mesh_by_regionid(
            dpoints, dvectors, dradii, i_vessel, slice_index, invert_status, first_clip, stl_surf_m,plot_flag)

        toc = time.perf_counter()
        time_minutes = (toc-tic)/60
        
        branch_centers = branch_mesh.cell_centers()
        branch_center_points = branch_centers.points
        branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
        branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

        # plot = pv.Plotter()
        # plot.add_mesh(branch_segment_extracted)
        # plot.add_mesh(stl_surf_m,opacity=0.3)
        # plot.show() 

        # Store branchid in dictionary
        dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
        dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
        
        print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")


#%% Slice vessels that require 2 slices

two_slice_vessels = ['L_P1'] #'
anterior_two_slice_vessels = ['L_A1','R_A1']

for vessel_name in two_slice_vessels:
    # Check that the collateral pathway is present
    if vessel_name in all_vessel_names:
        i_vessel = all_vessel_names.index(vessel_name)
        
        # Info for first slice
        slice_index = -1 
        invert_status = False
        
        # Store slice index for first slice
        dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
        
        tic = time.perf_counter()
        if vessel_name in anterior_two_slice_vessels:
            first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,anterior,stl_surf_m,plot_flag)
            # Reverse direction of the slice
            if slice_index == 0:
                slice_index = -1
                invert_status = False
            else:
                slice_index = 0
                invert_status = True
            branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
        else:
            first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,posterior,stl_surf_m,plot_flag)
            # Reverse direction of the slice
            if slice_index == 0:
                slice_index = -1
                invert_status = False
            else:
                slice_index = 0
                invert_status = True

            branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m,plot_flag)
        
        toc = time.perf_counter()
        time_minutes = (toc-tic)/60
        
        branch_centers = branch_mesh.cell_centers()
        branch_center_points = branch_centers.points
        branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
        branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)


        # plot = pv.Plotter()
        # plot.add_mesh(branch_segment_extracted)
        # plot.add_mesh(stl_surf_m,opacity=0.3)
        # plot.show() 

        # Store branchid in dictionary
        dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
        dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
        
        print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")        
        
    
#%%

# #%% Pcoms that require 2 slices

# Pcom_vessels = ['L_PCOM']

# for vessel_name in Pcom_vessels:
#     # Check that the collateral pathway is present
#     if vessel_name in all_vessel_names:
#         i_vessel = all_vessel_names.index(vessel_name)

#         # Info for first slice
#         slice_index = -1 #0
#         invert_status = False #True

#         # Store slice index for first slice
#         dsliceindex["sliceindex{}".format(
#             i_vessel)] = vessel_name, slice_index, invert_status

#         tic = time.perf_counter()

#         first_clip = clip_mesh_by_largest(
#               dpoints, dvectors, dradii, i_vessel, slice_index, invert_status, left, stl_surf_m, plot_flag) #left,
        
#         # Reverse direction of the slice
#         if slice_index == 0:
#             slice_index = -1
#             invert_status = False
#         else:
#             slice_index = 0
#             invert_status = True
            
#         branch_mesh, branch_id = clip_mesh_by_regionid(
#             dpoints, dvectors, dradii, i_vessel, slice_index, invert_status, first_clip, stl_surf_m,plot_flag)

#         toc = time.perf_counter()
#         time_minutes = (toc-tic)/60
        
#         branch_centers = branch_mesh.cell_centers()
#         branch_center_points = branch_centers.points
#         branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
#         branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

#         # plot = pv.Plotter()
#         # plot.add_mesh(branch_segment_extracted)
#         # plot.add_mesh(stl_surf_m,opacity=0.3)
#         # plot.show() 

#         # Store branchid in dictionary
#         dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
#         dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
        
#         print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")




#%%


# #%% Right
# tic = time.perf_counter()
# invert_status = False
# right_geom_data_filename = geometry_path + 'right.dat'    
# right = clip_by_dat_file(right_geom_data_filename,cow_only,invert_status,stl_surf_m,plot_flag)

# right_centers = right.cell_centers()
# right_center_points = right.points
# right_center_point_indices = mesh_data_final.find_containing_cell(right_center_points)
# right_segment_extracted = mesh_data_final.extract_cells(right_center_point_indices)

# i_store = num_vessels+3
# dcenterindices["indices{}".format(i_store)] = 'right', right_center_point_indices    

# toc = time.perf_counter()
# time_minutes = (toc-tic)/60
# print(f"Right slices  took {time_minutes:0.4f} minutes")



#%%

# Pcom_vessels = ['L_PCOM']

# for vessel_name in Pcom_vessels:
#     # Check that the collateral pathway is present
#     if vessel_name in all_vessel_names:
#         i_vessel = all_vessel_names.index(vessel_name)

#         # Info for first slice
#         slice_index = 0
#         invert_status = True

#         # Store slice index for first slice
#         dsliceindex["sliceindex{}".format(
#             i_vessel)] = vessel_name, slice_index, invert_status

#         tic = time.perf_counter()

#         first_clip = clip_mesh_by_largest(
#               dpoints, dvectors, dradii, i_vessel, slice_index, invert_status, cow_only, stl_surf_m,plot_flag)
        
#         # Reverse direction of the slice
#         if slice_index == 0:
#             slice_index = -1
#             invert_status = False
#         else:
#             slice_index = 0
#             invert_status = True
            
#         branch_mesh, branch_id = clip_mesh_by_regionid(
#             dpoints, dvectors, dradii, i_vessel, slice_index, invert_status, first_clip, stl_surf_m,plot_flag)

#         toc = time.perf_counter()
#         time_minutes = (toc-tic)/60
        
#         branch_centers = branch_mesh.cell_centers()
#         branch_center_points = branch_centers.points
#         branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
#         branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

#         # plot = pv.Plotter()
#         # plot.add_mesh(branch_segment_extracted)
#         # plot.add_mesh(stl_surf_m,opacity=0.3)
#         # plot.show() 

#         # Store branchid in dictionary
#         dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
#         dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices
        
#         print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")


#%%

# #%% Slice right Pcom with slice_rpcom.dat

# # Clip with .dat file
# invert_status = False
# vessel_name = "R_PCOM"
# slice_dat_filename = geometry_path + 'slice_rpcom.dat'

# tic = time.perf_counter()

# slice_index = 0
# invert_status = True

# first_clip = clip_by_dat_file(slice_dat_filename,cow_only,invert_status,stl_surf_m,plot_flag)


# # # Reverse direction of the slice
# # if slice_index == 0:
# #     slice_index = -1
# #     invert_status = False
# # else:
# slice_index = 0
# invert_status = True
# branch_mesh, branch_id = clip_mesh_by_regionid(
#     dpoints, dvectors, dradii, i_vessel, slice_index, invert_status, first_clip, stl_surf_m,plot_flag)

# toc = time.perf_counter()
# time_minutes = (toc-tic)/60


# branch_centers = branch_mesh.cell_centers()
# branch_center_points = branch_centers.points
# branch_center_point_indices = mesh_data_final.find_containing_cell(branch_center_points)
# branch_segment_extracted = mesh_data_final.extract_cells(branch_center_point_indices)

# # plot = pv.Plotter()
# # plot.add_mesh(branch_segment_extracted)
# # plot.add_mesh(stl_surf_m,opacity=0.3)
# # plot.show() 

# # Store branchid in dictionary
# dbranchid["branchid{}".format(i_vessel)] = vessel_name, branch_id
# dcenterindices["indices{}".format(i_vessel)] = vessel_name, branch_center_point_indices

# print(f"{vessel_name:s} slice  took {time_minutes:0.4f} minutes")



#%%

# #%% Slice CoW with basilar.dat

# # Clip with .dat file
# invert_status = False
# vessel_name = 'basilar'
# slice_dat_filename = geometry_path + vessel_name.lower() + '.dat'
# cow_only = clip_by_dat_file(slice_dat_filename,mesh_data_final,invert_status,stl_surf_m,plot_flag)

# cow_centers = cow_only.cell_centers()
# cow_center_points = cow_centers.points
# cow_center_point_indices = mesh_data_final.find_containing_cell(cow_center_points)
# cow_segment_extracted = mesh_data_final.extract_cells(cow_center_point_indices)

# # plot = pv.Plotter()
# # plot.add_mesh(cow_segment_extracted)
# # plot.add_mesh(stl_surf_m,opacity=0.3)
# # plot.show()

# mid_slice_index = 0

# #% Store slice index and invert status in dictionary
# i_store = num_vessels
# dsliceindex["sliceindex{}".format(i_store)] = "cow", mid_slice_index, invert_status
# dcenterindices["indices{}".format(i_store)] = 'cow', cow_center_point_indices

# print(f"Isolated CoW took {time_minutes:0.4f} minutes")


