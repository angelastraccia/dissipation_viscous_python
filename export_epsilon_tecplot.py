import tecplot as tp
from tecplot.exception import *
from tecplot.constant import *

from tqdm import tqdm
import glob
import os
import time

#%%

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
#%%

# pinfo = input('Patient -- ')
# case = input('Condition -- ')


def main(pinfo,case):
    num_cycle = 2
    onlyfiles, dat_indices, pathwd = get_list_files_dat(pinfo,case,num_cycle)
    cas_filename = 'L:\\vasospasm\\'+pinfo+'\\'+case+'\\3-computational\\hyak_submit\\'+pinfo+'_'+case+'.cas'
    print(cas_filename)
    
    tic = time.perf_counter()
    
    dissipation_viscous_directory = 'L:\\vasospasm\\'+pinfo+'\\'+case+'\\4-results\\dissipation_viscous\\'
    
    if not os.path.exists(dissipation_viscous_directory):
        os.makedirs(dissipation_viscous_directory)
    
    # Uncomment the following line to connect to a running instance of Tecplot 360:
    tp.session.connect()
    
    for i_data in tqdm(range(0,30)):
    
    
        dat_filename = 'L:\\vasospasm\\'+pinfo+'\\'+case+'\\3-computational\\hyak_submit\\'+dat_indices[i_data]+'.dat'
        ascii_filename = 'L:\\vasospasm\\'+pinfo+'\\'+case+'\\4-results\\dissipation_viscous\\'+ dat_indices[i_data] + '_epsilon.dat'
    
        # Load data
        dataset = tp.data.load_fluent(case_filenames=[cas_filename],
            data_filenames=[dat_filename],
            append=False)
        
        # Delete walls
        delete_zone_indices = range(1,dataset.num_zones)
        tp.active_frame().dataset.delete_zones([delete_zone_indices])
        
        # Define viscous dissipation variable
        tp.macro.execute_extended_command(command_processor_id='CFDAnalyzer4',
            command="Calculate Function='VELOCITYGRADIENT' Normalization='None' ValueLocation='CellCentered' CalculateOnDemand='T' UseMorePointsForFEGradientCalculations='F'")
        tp.data.operate.execute_equation(equation='{s11} = {dUdX}',
            value_location=ValueLocation.CellCentered)
        tp.data.operate.execute_equation(equation='{s12} = 0.5*({dUdY}+{dVdX})',
            value_location=ValueLocation.CellCentered)
        tp.data.operate.execute_equation(equation='{s13} = 0.5*({dUdZ}+{dWdX})',
            value_location=ValueLocation.CellCentered)
        tp.data.operate.execute_equation(equation='{s22} = {dVdY}',
            value_location=ValueLocation.CellCentered)
        tp.data.operate.execute_equation(equation='{s23} = 0.5*({dVdZ}+{dWdY})',
            value_location=ValueLocation.CellCentered)
        tp.data.operate.execute_equation(equation='{s33} = {dWdZ}',
            value_location=ValueLocation.CellCentered)
        tp.data.operate.execute_equation(equation='{epsilon} = 2*0.0035*(2*({s12}**2 + {s13}**2 + {s23}**2) + {s11}**2 + {s22}**2 + {s33}**2)',
            value_location=ValueLocation.CellCentered)
        
        # Export data to ASCII file
        tp.data.save_tecplot_ascii(ascii_filename,
            zones=[0],
            variables=[0,1,2,59],
            include_text=False,
            precision=9,
            include_geom=False,
            include_data_share_linkage=True)
    
    toc = time.perf_counter()
    time_minutes = (toc-tic)/60
    print(f"Tecplot ASCII export took {time_minutes:0.4f} minutes")


main('vsp19','baseline')