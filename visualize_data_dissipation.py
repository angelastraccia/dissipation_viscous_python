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

def calculate_dissipation(pinfo, case, i_vessel, ddissipation):

    len_cycle = 30
    num_cycle = 2
    onlydat, data_indices, pathwd = get_list_files_dat(pinfo, case, num_cycle)
    
    dissipation_in_time = [ddissipation.get("{}".format(data_indices[k])).get("dissipation{}".format(i_vessel))[1]
        for k in range(len_cycle)]
    
    # Dissipation in micro watts
    dissipation = np.mean(dissipation_in_time)*1e6
    
    # Volume in mm^2
    volume = ddissipation.get("{}".format(data_indices[0])).get(
        "dissipation{}".format(i_vessel))[2]*1e6
    
    dissipation_per_volume_in_time = [ddissipation.get("{}".format(data_indices[k])).get(
        "dissipation{}".format(i_vessel))[3]
        for k in range(len_cycle)]

    # Dissipation per volume in W/m^2
    dissipation_per_volume = np.mean(dissipation_per_volume_in_time)

    return dissipation, volume, dissipation_per_volume

#%% Load data

pinfo = input('Patient number -- ')
num_cycle = 2

case = 'baseline'
dissipation_path_baseline = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/dissipation_viscous/"
resistance_path_baseline = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/pressure_resistance/"
dpoints_bas = load_dict(resistance_path_baseline + "points_" + pinfo + "_" + case)
ddissipation_bas = load_dict(dissipation_path_baseline + "dissipation_" + pinfo + "_" + case)

case = 'vasospasm'
dissipation_path_vasospasm = "L:/vasospasm/" + pinfo + "/" + case + "/4-results/dissipation_viscous/"
ddissipation_vas = load_dict(dissipation_path_vasospasm + "dissipation_" + pinfo + "_" + case)

#%% Remove existing directories and create new ones for figures
# Save figures in both 4-results baseline and vasospasm directories
figure_path_baseline = (dissipation_path_baseline+ "/figures/")
figure_path_vasospasm = (dissipation_path_vasospasm+ "/figures/")

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

num_vessels = len(dpoints_bas)

# Define a color map for percent difference in resistance
cmap = cm.get_cmap('RdPu')
N_colors = 10
color_range = cmap(np.linspace(0,1,N_colors))
percent_diff_min = 0
percent_diff_max = 2000
dissipation_percent_difference_range = np.linspace(percent_diff_min,percent_diff_max,N_colors)

# Instantiate variables
df_all_vessels = pd.DataFrame()
df_colors_all_vessels = pd.DataFrame()
dresistance_bas, dresistance_vas = {}, {}

for i_vessel in range(num_vessels):
    
    vessel_name = dpoints_bas.get("points{}".format(i_vessel))[0]
    
    dissipation_bas, volume_bas, dissipation_per_volume_bas = calculate_dissipation(pinfo, 'baseline', i_vessel, ddissipation_bas)
    dissipation_vas, volume_vas, dissipation_per_volume_vas = calculate_dissipation(pinfo, 'vasospasm', i_vessel, ddissipation_vas)
    
    percent_diff_dissipation = (dissipation_vas-dissipation_bas)/dissipation_bas*100
    percent_diff_volume = (volume_vas-volume_bas)/volume_bas*100
    percent_diff_dissipation_per_volume = (dissipation_per_volume_vas-dissipation_per_volume_bas)/dissipation_per_volume_bas*100

    vessel_data = {
        'viscous dissipation baseline': dissipation_bas,
        'viscous dissipation vasospasm': dissipation_vas,
        'viscous dissipation percent difference': percent_diff_dissipation,
        'volume baseline': volume_bas,
        'volume vasospasm': volume_vas,
        'volume percent difference': percent_diff_volume,
        'dissipation per volume baseline': dissipation_per_volume_bas,
        'dissipation per volume vasospasm': dissipation_per_volume_vas,
        'dissipation per volume percent difference': percent_diff_dissipation_per_volume,
    }

    df_vessel = pd.DataFrame(vessel_data, index=[vessel_name])
    #df_vessel.loc()
    df_all_vessels = pd.concat([df_all_vessels,df_vessel])
    
    # Determine which color is associated with that percent change
    percent_change_difference = np.abs(
        np.subtract(dissipation_percent_difference_range, percent_diff_dissipation_per_volume))
    closest_color_index = np.argmin(percent_change_difference)
    
    # Convet it to the RGB value from [0,255]
    color_vessel = np.round(np.multiply(color_range[closest_color_index,0:3],255))
    
    # Save color in data frame
    color_data = {vessel_name: color_vessel}
    
    df_color_vessel = pd.DataFrame(color_data, index=['red','green','blue'])
    #df_color_vessel.loc()
    df_colors_all_vessels = pd.concat([df_colors_all_vessels, df_color_vessel],axis=1)

#% Plot heat map

f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize = (25,17))

plt.suptitle("Dissipation heatmap for "+ pinfo, fontsize = 40)
sns.set(font_scale=1.8)

xlabels = ['baseline','vasospasm']

sns.heatmap(df_all_vessels.loc[:,['viscous dissipation baseline','viscous dissipation vasospasm']],xticklabels=xlabels,annot = True,cmap =plt.cm.RdPu,fmt = '.0f',linewidth=0.5,ax=ax1)
ax1.set_title('Dissipation [microW]',fontsize=30)   

sns.heatmap(df_all_vessels.loc[:,['volume baseline','volume vasospasm']],xticklabels=xlabels,annot = True,cmap =plt.cm.RdPu,fmt = '.2f',linewidth=0.5,ax=ax2)
ax2.set_yticks([])
ax2.set_title('Volume [mm^3]',fontsize=30)

sns.heatmap(df_all_vessels.loc[:,['dissipation per volume baseline','dissipation per volume vasospasm']],xticklabels=xlabels,annot = True,cmap =plt.cm.RdPu,linewidth=0.5,ax=ax3)
ax3.set_yticks([])
ax3.set_title('Dissipation per volume [W/m^3]',fontsize=30)

sns.heatmap(df_all_vessels.loc[:,['viscous dissipation percent difference','volume percent difference','dissipation per volume percent difference']],xticklabels=['Dissipation','Volume','Dissipation per volume'],annot = True,fmt ='.0f',cmap =plt.cm.RdPu,linewidth=0.5,ax=ax4,vmin=percent_diff_min, vmax=percent_diff_max)
for t in ax4.texts: t.set_text(t.get_text() + " %")
ax4.set_title('Percent difference',fontsize=30)
ax4.set_yticks([])

plt.savefig(figure_path_baseline + "plot_heatmap_dissipation_threshold_" + str(percent_diff_max) + "_" + pinfo + ".png")
plt.savefig(figure_path_vasospasm + "plot_heatmap_dissipation_threshold_" + str(percent_diff_max) + "_" + pinfo + ".png")

# Export color data for percent change in resistance to CSV file
df_colors_all_vessels.to_csv(dissipation_path_baseline + pinfo + "_colors_dissipation_threshold_" + str(percent_diff_max) + ".csv")
df_colors_all_vessels.to_csv(dissipation_path_vasospasm + pinfo + "_colors_dissipation_threshold_" + str(percent_diff_max) + ".csv")

#%% Write data to CSV

vessel_list = ["L_MCA","R_MCA","L_A2","R_A2","L_P2","R_P2","L_TICA","R_TICA","BAS",
               "L_A1","R_A1","L_PCOM","R_PCOM","L_P1","R_P1"]


df_percent_diff_all_vessels = pd.DataFrame()

for vessel_of_interest in vessel_list:

    # Create data frame with all vessels and percent change
    
    if vessel_of_interest in df_all_vessels.index:
        
        vessel_percent_diff_data = {vessel_of_interest: df_all_vessels.loc[vessel_of_interest,"dissipation per volume percent difference"]}
        
        print(df_all_vessels.loc[vessel_of_interest,"dissipation per volume percent difference"])
         
    else:
        vessel_percent_diff_data = {vessel_of_interest: 'nan'}
        print('missing')
        
    df_percent_diff_vessel =  pd.DataFrame(vessel_percent_diff_data, index=[pinfo])
    
    df_percent_diff_all_vessels = pd.concat([df_percent_diff_all_vessels, df_percent_diff_vessel],axis=1)


df_percent_diff_all_vessels.to_csv(dissipation_path_vasospasm + pinfo + "_dissipation_percent_difference.csv")
