# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:45:11 2022

@author: GALADRIEL_GUEST
"""

import scipy.io
import scipy
import math
import pandas as pd
import glob
import csv
import numpy as np
import pickle
import importlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import seaborn as sns
import shutil
import pyvista as pv

os.chdir("L:/vasospasm/calculation_resistance_dissipation/dissipation_viscous_python")

pd.options.display.float_format = '{:.2f}'.format


# %% Functions & Main


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

def calculate_dissipation_vs_subzone(pinfo, case, i_vessel, ddissipation):

    len_cycle = 30
    num_cycle = 2
    onlydat, data_indices, pathwd = get_list_files_dat(pinfo, case, num_cycle)
    
    num_subzones = len(ddissipation.get("{}".format(data_indices[0])).get("dissipation{}".format(i_vessel))[1])
    
    # Volume in mm^2
    volume = ddissipation.get("{}".format(data_indices[0])).get(
        "dissipation{}".format(i_vessel))[2]*1e6
    
    integrated_dissipation_in_time = np.array([ddissipation.get("{}".format(data_indices[k])).get("dissipation{}".format(i_vessel))[1]
        for k in range(len_cycle)])
    
    integrated_dissipation = np.zeros((3,num_subzones))
    
    # Dissipation integrated over volume in W*mm^3
    integrated_dissipation[0,:] = np.min(integrated_dissipation_in_time,axis=0)*1e6
    integrated_dissipation[1,:] = np.mean(integrated_dissipation_in_time,axis=0)*1e6
    integrated_dissipation[2,:] = np.max(integrated_dissipation_in_time,axis=0)*1e6
    
    dissipation_in_time = np.array([ddissipation.get("{}".format(data_indices[k])).get(
        "dissipation{}".format(i_vessel))[3]
        for k in range(len_cycle)])

    # Dissipation in W
    dissipation = np.zeros((3,num_subzones))
    
    dissipation[0,:] = np.min(dissipation_in_time,axis=0)
    dissipation[1,:] = np.mean(dissipation_in_time,axis=0)
    dissipation[2,:] = np.max(dissipation_in_time,axis=0)

    return integrated_dissipation, volume, dissipation

def plot_volume_dissipation_vs_subzone(integrated_dissipation, volume, dissipation, case, ax1, ax2, ax3, ax4):
    
    dist = np.linspace(1,np.shape(volume)[0],np.shape(volume)[0])
    
    # Plot volume
    ax1.plot(dist,volume,label=case)
    
    # Plot integrated dissipation
    ax2.plot(dist,integrated_dissipation[1,:], "-",label=case)
    #ax2.fill_between(dist,integrated_dissipation[0,:],integrated_dissipation[2,:], alpha=0.2)
    
    # Plot dissipation
    ax3.plot(dist,dissipation[1,:], "-",label=case)
    #ax3.fill_between(dist,dissipation[0,:],dissipation[2,:], alpha=0.2)
    
    dissipation_total = dissipation[1,:].sum()
    
    percent_of_total_dissipation = np.divide(dissipation[1,:],dissipation_total)*100
    
    ax4.plot(dist,percent_of_total_dissipation,'-',label=case)


    return ax1, ax2, ax3, ax4

def plot_subzones_pyvista(dcenterpoints_subzones, mesh_data_final,stl_surf):
    cmap = plt.get_cmap('inferno')

    num_vessels = len(dcenterpoints_subzones)
    
    p = pv.Plotter()
    p.add_mesh(stl_surf,opacity = 0.2,color="white")
    
    for i_vessel in range(0,num_vessels):
        num_subzones = len(dcenterpoints_subzones.get("indices{}".format(i_vessel))) - 1
        
        # Define a discretized color map
        colors = cmap(np.linspace(0, .8, num_subzones+1))
        
        for i_subzone in range(1,num_subzones+1):
            
            vessel_subzone_center_point_indices = dcenterpoints_subzones.get("indices{}".format(i_vessel))[i_subzone]
            vessel_subzone_segment_extracted = mesh_data_final.extract_cells(vessel_subzone_center_point_indices)
            
            p.add_mesh(vessel_subzone_segment_extracted,color=colors[i_subzone])
                         
    p.show()
    
    return 
#%% Load data

pinfo = 'pt40' #input('Patient number -- ')
num_cycle = 2
i_data = 0
plot_subzones_flag = 0

case = 'baseline'
dissipation_path_baseline = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/dissipation_viscous/"
resistance_path_baseline = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/pressure_resistance/"
dpoints_bas = load_dict(resistance_path_baseline + "points_" + pinfo + "_" + case)
ddissipation_bas = load_dict(dissipation_path_baseline + "dissipation_subzones_" + pinfo + "_" + case)
dcenterpoints_subzones_bas = load_dict(dissipation_path_baseline + "centerpoint_indices_vessel_subzones_" + pinfo + "_" + case) 

case = 'vasospasm'
dissipation_path_vasospasm = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/dissipation_viscous/"
resistance_path_vasospasm = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/pressure_resistance/"
dpoints_vas = load_dict(resistance_path_vasospasm + "points_" + pinfo + "_" + case)
ddissipation_vas = load_dict(dissipation_path_vasospasm + "dissipation_subzones_" + pinfo + "_" + case)
dcenterpoints_subzones_vas = load_dict(dissipation_path_vasospasm + "centerpoint_indices_vessel_subzones_" + pinfo + "_" + case) 


if plot_subzones_flag == 1:

    # Read in Tecplot data file
    # Get list of filenames
    onlyfiles, data_indices, pathwd = get_list_files_dat(pinfo,"baseline",num_cycle)
    data_filename = dissipation_path_baseline + data_indices[i_data] + '_epsilon.dat'
    reader = pv.get_reader(data_filename)
    mesh_data_imported_bas = reader.read()
    mesh_data_final_bas = mesh_data_imported_bas[0]
    print(mesh_data_final_bas.cell_data)
    
    onlyfiles, data_indices, pathwd = get_list_files_dat(pinfo,"vasospasm",num_cycle)
    data_filename = dissipation_path_vasospasm + data_indices[i_data] + '_epsilon.dat'
    reader = pv.get_reader(data_filename)
    mesh_data_imported_vas = reader.read()
    mesh_data_final_vas = mesh_data_imported_vas[0]
    print(mesh_data_final_vas.cell_data)
    
    # Read in STL of surface
    fname_stl_bas = "L:/vasospasm/" + pinfo + "/" + case + "/1-geometry/" + pinfo + '_' + case + '_final.stl'
    stl_surf_bas = pv.read(fname_stl_bas)
    fname_stl_vas = "L:/vasospasm/" + pinfo + "/" + case + "/1-geometry/" + pinfo + '_' + case + '_final.stl'
    stl_surf_vas = pv.read(fname_stl_vas)
    
    # # Plot subzones
    
    # plot_subzones_pyvista(dcenterpoints_subzones_bas, mesh_data_final_bas,stl_surf_bas)
    # plot_subzones_pyvista(dcenterpoints_subzones_vas, mesh_data_final_vas,stl_surf_vas)

#%% Remove existing directories and create new ones for figures
# Save figures in both 4-results baseline and vasospasm directories
figure_path_baseline = (dissipation_path_baseline+ "/figures_subzones/")
figure_path_vasospasm = (dissipation_path_vasospasm + "/figures_subzones/")

# Save figures in both 4-results baseline and vasospasm directories
if os.path.exists(figure_path_baseline):
    shutil.rmtree(figure_path_baseline)
if os.path.exists(figure_path_vasospasm):
    shutil.rmtree(figure_path_vasospasm)

if not os.path.exists(figure_path_baseline):
    os.makedirs(figure_path_baseline)
if not os.path.exists(figure_path_vasospasm):
    os.makedirs(figure_path_vasospasm)

   

#%%

# Define a color map for percent difference in dissipation
cmap = cm.get_cmap('RdPu')
percent_diff_min = 0
percent_diff_max = 1000

# Instantiate variables
df_all_vessels = pd.DataFrame()

num_vessels = len(dcenterpoints_subzones_bas)

for i_vessel in range(num_vessels):
    
    vessel_name = dpoints_bas.get("points{}".format(i_vessel))[0]
    
    integrated_dissipation_bas, volume_bas, dissipation_bas = calculate_dissipation_vs_subzone(pinfo, 'baseline', i_vessel, ddissipation_bas)
    integrated_dissipation_vas, volume_vas, dissipation_vas = calculate_dissipation_vs_subzone(pinfo, 'vasospasm', i_vessel, ddissipation_vas)
    
    integrated_dissipation_bas_total = integrated_dissipation_bas[1,:].sum()
    volume_bas_total = volume_bas.sum()
    dissipation_bas_total = dissipation_bas[1,:].sum()

    integrated_dissipation_vas_total = integrated_dissipation_vas[1,:].sum()
    volume_vas_total = volume_vas.sum()
    dissipation_vas_total = dissipation_vas[1,:].sum()
    
    percent_diff_integrated_dissipation = (integrated_dissipation_vas_total-integrated_dissipation_bas_total)/integrated_dissipation_bas_total*100
    percent_diff_volume = (volume_vas_total-volume_bas_total)/volume_bas_total*100
    percent_diff_dissipation = (dissipation_vas_total-dissipation_bas_total)/dissipation_bas_total*100

    vessel_data = {
        'viscous dissipation baseline': integrated_dissipation_bas_total,
        'viscous dissipation vasospasm': integrated_dissipation_vas_total,
        'viscous dissipation percent difference': percent_diff_integrated_dissipation,
        'volume baseline': volume_bas_total,
        'volume vasospasm': volume_vas_total,
        'volume percent difference': percent_diff_volume,
        'dissipation per volume baseline': dissipation_bas_total,
        'dissipation per volume vasospasm': dissipation_vas_total,
        'dissipation per volume percent difference': percent_diff_dissipation,
    }

    df_vessel = pd.DataFrame(vessel_data, index=[vessel_name])
    #df_vessel.loc()
    df_all_vessels = pd.concat([df_all_vessels,df_vessel])
    
    # % Create a plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 25))
    plt.rcParams["axes.grid"] = True
    plt.rcParams.update({"grid.color": "0.5","grid.linestyle": "-","grid.linewidth": 0.5})

    plot_volume_dissipation_vs_subzone(integrated_dissipation_bas, volume_bas, dissipation_bas, "baseline", ax1, ax2, ax3, ax4)
    plot_volume_dissipation_vs_subzone(integrated_dissipation_vas, volume_vas, dissipation_vas, "vasospasm", ax1, ax2, ax3, ax4)

    fig.suptitle(vessel_name,fontsize = 40)
    subtitle_fontsize = 30

    ax1.set_ylim([0, None])
    ax1.set_title("Volume [mm^3]",fontsize = subtitle_fontsize)
    ax1.legend(fontsize="large",facecolor="white")
    ax1.set_facecolor('white')
    
    ax2.set_ylim([0, None])
    ax2.set_title("Integrated dissipation [W*mm^3]",fontsize = subtitle_fontsize)
    ax2.legend(fontsize="large",facecolor="white")
    ax2.set_facecolor('white')
    
    ax3.set_ylim([0, None])
    ax3.set_title("Dissipation [W]",fontsize = subtitle_fontsize)
    ax3.legend(fontsize="large",facecolor="white")
    ax3.set_facecolor('white')
    
    ax4.set_title("Percent of total dissipation [%]",fontsize = subtitle_fontsize)
    ax4.legend(fontsize="large",facecolor="white")
    ax4.set_facecolor('white')
    ax4.set_ylim([0, 50])
    
    plt.savefig(figure_path_baseline + "/dissipation_subzones_" + vessel_name+ ".png")
    plt.savefig(figure_path_vasospasm + "/dissipation_subzones_"  + vessel_name + ".png")
    

#%% Plot heat map

f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize = (25,17))

plt.suptitle("Dissipation heatmap for "+ pinfo, fontsize = 40)
sns.set(font_scale=1.8)

xlabels = ['baseline','vasospasm']

sns.heatmap(df_all_vessels.loc[:,['viscous dissipation baseline','viscous dissipation vasospasm']],xticklabels=xlabels,annot = True,cmap =plt.cm.RdPu,fmt = '.0f',linewidth=0.5,ax=ax1)
ax1.set_title('Dissipation [W*mm^3]',fontsize=30)   

sns.heatmap(df_all_vessels.loc[:,['volume baseline','volume vasospasm']],xticklabels=xlabels,annot = True,cmap =plt.cm.RdPu,fmt = '.2f',linewidth=0.5,ax=ax2)
ax2.set_yticks([])
ax2.set_title('Volume [mm^3]',fontsize=30)

sns.heatmap(df_all_vessels.loc[:,['dissipation per volume baseline','dissipation per volume vasospasm']],xticklabels=xlabels,annot = True,cmap =plt.cm.RdPu,linewidth=0.5,ax=ax3)
ax3.set_yticks([])
ax3.set_title('Dissipation [W]',fontsize=30)

sns.heatmap(df_all_vessels.loc[:,['viscous dissipation percent difference','volume percent difference','dissipation per volume percent difference']],xticklabels=['Dissipation','Volume','Dissipation per volume'],annot = True,fmt ='.0f',cmap =plt.cm.RdPu,linewidth=0.5,ax=ax4,vmin=percent_diff_min, vmax=percent_diff_max)
for t in ax4.texts: t.set_text(t.get_text() + " %")
ax4.set_title('Percent difference',fontsize=30)
ax4.set_yticks([])

plt.savefig(figure_path_baseline + "plot_heatmap_dissipation_threshold_" + str(percent_diff_max) + "_" + pinfo + ".png")
plt.savefig(figure_path_vasospasm + "plot_heatmap_dissipation_threshold_" + str(percent_diff_max) + "_" + pinfo + ".png")





