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

def get_list_files_dat(pinfo, case, base_size, num_cycle):
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

    pathwd = "N:/vasospasm/mesh_independence_study/" + pinfo + "/" + case + "/base_size_" + base_size + "/3-computational/hyak_submit/"

    os.chdir(pathwd)
    onlyfiles = []
    for file in glob.glob("*.dat"):
        if pinfo + "_" + case + "_cycle" + num_cycle in file:
            onlyfiles.append(file)
    indices = [l[:-4] for l in onlyfiles]

    return onlyfiles, indices,pathwd

#%%

def clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,mesh_data,stl_surf_m):
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
    
    # Plot clipped surface
    p = pv.Plotter()
    p.add_mesh(mesh_clipped_connected, scalars='RegionId',label=vessel_name)
    p.add_legend(size=(.2, .2), loc='upper right')
    p.add_mesh(stl_surf_m, opacity=0.3)
    p.show()
    
    branch_id = int(input('Which branch id? -- '))

    # Identify which cells are associated with the branch
    clipped_cells = np.where(mesh_clipped_connected['RegionId']==branch_id)[0]
    clipped_volume = mesh_clipped_connected.extract_cells(clipped_cells)

    p = pv.Plotter()
    p.add_mesh(stl_surf_m, opacity=0.3)
    p.add_mesh(clipped_volume,label=vessel_name)
    p.add_legend(size=(.2, .2), loc='upper right')
    p.show()
    
    return clipped_volume, branch_id

def clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,mesh_data,stl_surf_m):
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
    
    p = pv.Plotter()
    p.add_mesh(stl_surf_m, opacity=0.3)
    p.add_mesh(largest_region, scalars='RegionId')
    p.show()
    
    return largest_region

def clip_by_dat_file(dat_filename,mesh_data,invert_status,stl_surf_m):
    
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

    p = pv.Plotter()
    p.add_mesh(stl_surf_m, opacity=0.3)
    p.add_mesh(largest_region, scalars='RegionId')
    p.show()

    return largest_region

def clip_mesh(dat_filename,mesh_data,invert_status,stl_surf_m):
    
    geom_data = np.genfromtxt(dat_filename,
                         skip_header=1)
    plane_center = (geom_data[0],geom_data[1],geom_data[2])
    plane_normal = (geom_data[3],geom_data[4],geom_data[5])
    slice_plane = pv.Plane(center = plane_center, direction = plane_normal, i_size = 0.1, j_size = 0.1)

    # Clip data
    mesh_clipped = mesh_data.clip_surface(slice_plane, invert=invert_status)

    p = pv.Plotter()
    p.add_mesh(stl_surf_m, opacity=0.3)
    p.add_mesh(mesh_clipped)
    p.show()
    
    return mesh_clipped

#%%

# Define patient info
pinfo = input('Patient number -- ')
case = input('Condition -- ')
base_size = input('Base size -- ')

num_cycle = 2

# Identifies the directory with the case_info.mat file
dinfo = scipy.io.loadmat(
    "N:/vasospasm/mesh_independence_study/" + pinfo + "/" + case + "/case_info.mat"
)

# Extract the variation input number
variation_input_case_info = dinfo.get("variation_input")
variation_input = variation_input_case_info[0][0]

# Paths
geometry_path = "N:/vasospasm/mesh_independence_study/" + pinfo+"/" + case + "/1-geometry/"
results_path = "N:/vasospasm/mesh_independence_study/" + pinfo + "/" + case + "/base_size_" + base_size + "/4-results/dissipation_viscous/"

# Read in STL of surface
fname_stl_m = geometry_path + pinfo + '_' + case + '_final.stl'
stl_surf_m = pv.read(fname_stl_m)

# Read in centerlines, vectors, etc.
centerline_data_path = "N:/vasospasm/mesh_independence_study/" + pinfo + "/" + case + "/"
dvectors = load_dict(centerline_data_path +"vectors_" + pinfo + "_" + case)
dpoints = load_dict(centerline_data_path +"points_" + pinfo + "_" + case)
dradii= load_dict(centerline_data_path +"radii_" + pinfo + "_" + case)

# Define dictionary to store indices
dbranchid,dsliceindex,dcenterindices = {},{},{}

# Get list of filenames
i_data = 0
onlyfiles, data_indices, pathwd = get_list_files_dat(pinfo,case,base_size,num_cycle)
data_filename = results_path + data_indices[i_data] + '_epsilon.dat'

# data_filename = results_path + pinfo + '_' + case + '_epsilon.dat'

# Get list of vessel names
all_vessel_names = [dpoints.get("points{}".format(i_vessel))[0] for i_vessel in range(0,len(dpoints))]
num_vessels = len(all_vessel_names)

#%% Read in Tecplot data file
tic = time.perf_counter()
reader = pv.get_reader(data_filename)
mesh_data_imported = reader.read()
mesh_data_final = mesh_data_imported[0]
print(mesh_data_final.cell_data)
toc = time.perf_counter()
time_minutes = (toc-tic)/60
print(f"Data import took {time_minutes:0.4f} minutes")


#%% BAS
vessel_name = 'BAS'
BAS_vessel_index = all_vessel_names.index(vessel_name)

vessel_points = dpoints.get("points{}".format(BAS_vessel_index))[1]
vessel_vectors = dvectors.get("vectors{}".format(BAS_vessel_index))[1]

# Plot points and vectors
plot_vessel_points_vectors = pv.Plotter()
plot_points = pv.PolyData(vessel_points)
plot_vessel_points_vectors.add_mesh(stl_surf_m, opacity=0.3)
plot_vessel_points_vectors.add_mesh(plot_points)
plot_points["vectors"] = vessel_vectors*0.001
plot_points.set_active_vectors("vectors")
plot_vessel_points_vectors.add_mesh(
    plot_points.arrows, lighting=False)
plot_vessel_points_vectors.show()

# Determine which is the point associated with the first cut plane (beginning or end of points)
BAS_slice_index = int(input('Which point defines the first cut plane ? 0: first point, -1: last point -- '))
if BAS_slice_index == 0:
    invert_status = True
else:
    invert_status = False

tic = time.perf_counter()
BAS, BAS_branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,BAS_vessel_index,BAS_slice_index,invert_status,mesh_data_final,stl_surf_m)
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

#%% Isolate CoW

plot_vessel_points_vectors = pv.Plotter()
plot_points = pv.PolyData(vessel_points)
plot_vessel_points_vectors.add_mesh(stl_surf_m, opacity=0.3)
plot_vessel_points_vectors.add_mesh(plot_points)
plot_points["vectors"] = vessel_vectors*0.001
plot_points.set_active_vectors("vectors")
plot_vessel_points_vectors.add_mesh(
    plot_points.arrows, lighting=False)
plot_vessel_points_vectors.show()

# Switch the direction of the slicing compared to BAS slice
mid_slice_index = int(input('Which point defines the CoW slice? -- '))

if BAS_slice_index == 0:
    invert_status = False
else:
    invert_status = True

tic = time.perf_counter()
cow_only = clip_mesh_by_largest(dpoints,dvectors,dradii,BAS_vessel_index,mid_slice_index,invert_status,mesh_data_final,stl_surf_m)
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

#%% Split geometry into anterior and posterior

tic = time.perf_counter()
anterior_posterior_geom_data_filename = geometry_path + 'anterior_posterior.dat'

# Anterior
invert_status = False

# If the variation is missing a segment in the anterior circulation, keep all regions
if variation_input == 2 or variation_input == 5 or variation_input == 6:
    anterior = clip_mesh(anterior_posterior_geom_data_filename,cow_only,invert_status,stl_surf_m)
    
# Otherwise keep the largest region only
else:
    anterior = clip_by_dat_file(anterior_posterior_geom_data_filename,cow_only,invert_status,stl_surf_m)

anterior_centers = anterior.cell_centers()
anterior_center_points = anterior_centers.points
anterior_center_point_indices = mesh_data_final.find_containing_cell(anterior_center_points)
anterior_segment_extracted = mesh_data_final.extract_cells(anterior_center_point_indices)

# plot = pv.Plotter()
# plot.add_mesh(anterior_segment_extracted)
# plot.add_mesh(stl_surf_m,opacity=0.3)
# plot.show() 

i_store = num_vessels+1   
dcenterindices["indices{}".format(i_store)] = 'anterior', anterior_center_point_indices    

# Posterior
invert_status = True

# If the variation is missing a segment in the posterior circulation, keep all regions
if variation_input == 3 or variation_input == 4:
    posterior = clip_mesh(anterior_posterior_geom_data_filename,cow_only,invert_status,stl_surf_m)
# Otherwise keep the largest region only
else:
    posterior = clip_by_dat_file(anterior_posterior_geom_data_filename,cow_only,invert_status,stl_surf_m)

posterior_centers = posterior.cell_centers()
posterior_center_points = posterior_centers.points
posterior_center_point_indices = mesh_data_final.find_containing_cell(posterior_center_points)
posterior_segment_extracted = mesh_data_final.extract_cells(posterior_center_point_indices)

# plot = pv.Plotter()
# plot.add_mesh(posterior_segment_extracted)
# plot.add_mesh(stl_surf_m,opacity=0.3)
# plot.show() 

i_store = num_vessels+2 
dcenterindices["indices{}".format(i_store)] = 'posterior', posterior_center_point_indices

toc = time.perf_counter()
time_minutes = (toc-tic)/60
print(f"Anterior/posterior slices  took {time_minutes:0.4f} minutes")


#%% Slice vessels that require 1 slice

one_slice_vessels = ['L_MCA','R_MCA','L_A2','R_A2','L_P2','R_P2']  #
anterior_one_slice_vessels = ['L_MCA','R_MCA','L_A2','R_A2']
slice_index = 0
invert_status = True

for vessel_name in one_slice_vessels:
    i_vessel = all_vessel_names.index(vessel_name)
    
    tic = time.perf_counter()
    if vessel_name in anterior_one_slice_vessels:
        branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,anterior,stl_surf_m)
    else:
        branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,posterior,stl_surf_m)
    
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

two_slice_vessels = ['L_A1','R_A1','L_P1','R_P1'] #,
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
            first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,anterior,stl_surf_m)
            # Reverse direction of the slice
            if slice_index == 0:
                slice_index = -1
                invert_status = False
            else:
                slice_index = 0
                invert_status = True
            branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m)
        else:
            first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,posterior,stl_surf_m)
            # Reverse direction of the slice
            if slice_index == 0:
                slice_index = -1
                invert_status = False
            else:
                slice_index = 0
                invert_status = True
            branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m)
        
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

#%% Slice vessels that require 3 slices

# If missing a left Pcom, only slice with the .dat file on the right side
if variation_input == 7:
    three_slice_vessels = ['R_TICA']
    
    i_vessel = all_vessel_names.index(vessel_name)
    
    tic = time.perf_counter()
    # Clip using last point
    slice_index = -1
    invert_status = False
    first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,anterior,stl_surf_m)
    
    dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
    
    # Clip using first point
    slice_index = 0 
    invert_status = True
    branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m)
    
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


# If missing a right Pcom, only slice with the .dat file on the right side
elif variation_input == 8:
    three_slice_vessels = ['L_TICA']
    
    tic = time.perf_counter()
    # Clip using last point
    slice_index = -1
    invert_status = False
    first_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,anterior,stl_surf_m)
    
    dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
    
    # Clip using first point
    slice_index = 0 
    invert_status = True
    branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m)
    
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
    
# Otherwise slice with .dat files for both sides    
else:
    three_slice_vessels = ['L_TICA','R_TICA'] #

for vessel_name in three_slice_vessels:
    i_vessel = all_vessel_names.index(vessel_name)
    
    tic = time.perf_counter()
    slice_dat_filename = geometry_path + vessel_name.lower() + '.dat'
    
    # Clip with .dat file
    invert_status = False
    first_clip = clip_by_dat_file(slice_dat_filename,anterior,invert_status,stl_surf_m)
    
    # Clip using a point
    slice_index = 0
    invert_status = True
    second_clip = clip_mesh_by_largest(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,first_clip,stl_surf_m)
    
    dsliceindex["sliceindex{}".format(i_vessel)] = vessel_name, slice_index, invert_status
    
    # Reverse direction of the slice
    if slice_index == 0:
        slice_index = -1
        invert_status = False
    else:
        slice_index = 0
        invert_status = True
    branch_mesh, branch_id = clip_mesh_by_regionid(dpoints,dvectors,dradii,i_vessel,slice_index,invert_status,second_clip,stl_surf_m)
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

#%% Save dictionaries

save_dict(dbranchid, results_path +'branch_id_' + pinfo + '_' + case)
save_dict(dsliceindex, results_path +'slice_index_' + pinfo + '_' + case)
save_dict(dcenterindices, results_path +'centerpoint_indices_' + pinfo + '_' + case)

print('Saved dictionaries')



