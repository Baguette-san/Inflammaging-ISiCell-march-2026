from isicell import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import time
import psutil
import shutil
import os
import gc

# MAIN - Evaluations

def total_macrocircs_eval(df, col_macrocircs):
    return df[col_macrocircs].sum().sum()


def total_neutros_eval(df, col_neutros):
    return df[col_neutros].sum().sum()

    
def macrocircs_timing_eval(df, col_macrocircs):
    weights = df[col_macrocircs].sum(axis=1)
    return np.average(df["step"], weights=weights) if weights.sum() else -1

    
def neutros_timing_eval(df, col_neutros):
    weights = df[col_neutros].sum(axis=1)
    return np.average(df["step"], weights=weights) if weights.sum() else -1

    
def neutro_at_end_eval(df, col_neutros):
    return df[col_neutros].sum(axis=1).iloc[-1]

    
def apop_at_end_eval(df):
    return df["Apoptotic"].iloc[-1]

    
def necro_at_end_eval(df):
    return df["Necrotic"].iloc[-1]

    
def stromal_loss_eval(df, col_MSCs):
    return (df[col_MSCs].sum(axis=1).iloc[10]
            - df[col_MSCs].sum(axis=1).iloc[-1])

    
def macrores_loss_eval(df, col_macrores):
    return (df[col_macrores].sum(axis=1).iloc[10]
            - df[col_macrores].sum(axis=1).iloc[-1])
    

def local_cells_loss_eval(df, local_cells):
    return (df[local_cells].sum(axis=1).iloc[10]
            - df[local_cells].sum(axis=1).iloc[-1])


def infla_timing_eval(df):
    weights = df["INFLA"]
    return np.average(df["step"], weights=weights) if weights.sum() else -1

    
def reso_timing_eval(df):
    weights = df["RESO"]
    return np.average(df["step"], weights=weights) if weights.sum() else -1

    
def infla_at_end_eval(df):
    return df["INFLA"].iloc[-1]


# MAIN : Constraints from threshold

def circulant_recruited_test(total_macrocircs, total_neutros, total_macrocircs_thd=0, total_neutros_thd=0):
    return ((total_macrocircs > total_macrocircs_thd) & (total_neutros > total_neutros_thd))

def circulant_sequenced_test(macrocircs_timing, neutros_timing, macrocircs_neutros_delta_thd=0):
    return ((macrocircs_timing - neutros_timing) > macrocircs_neutros_delta_thd)

def neutros_peak_timing_test(neutros_timing, neutros_timing_range=(0,800)):
    return ((neutros_timing >= neutros_timing_range[0]) & (neutros_timing <= neutros_timing_range[1])) 

def neutro_cleared_test(neutro_at_end, neutro_at_end_thd=0):
    return (neutro_at_end <= neutro_at_end_thd)

def apop_cleared_test(apop_at_end, apop_at_end_thd=0):
    return (apop_at_end <= apop_at_end_thd)

def necro_cleared_test(necro_at_end, necro_at_end_thd=0):
    return (necro_at_end <= necro_at_end_thd)

def stromal_spared_test(stromal_loss, stromal_loss_thd=0):
    return (stromal_loss <= stromal_loss_thd)

def macrores_spared_test(macrores_loss, macrores_loss_thd=0):
    return (macrores_loss <= macrores_loss_thd)

def tissue_spared_test(local_cells_loss, local_cells_loss_thd=0):
    return (local_cells_loss <= local_cells_loss_thd)

def infla_timed_test(infla_timing, infla_timing_range=(0,800)):
    return ((infla_timing >= infla_timing_range[0]) & (infla_timing <= infla_timing_range[1]))

def reso_infla_sequenced_test(infla_timing, reso_timing, reso_infla_delta_timing_thd=0):
    return ((reso_timing - infla_timing) > reso_infla_delta_timing_thd)

def infla_cleared_test(infla_at_end, infla_at_end_thd=1):
    return (infla_at_end/1600 <= infla_at_end_thd)

# MAIN : apply evals and tests

col_neutros = ["Neutrophiles","NeutroInfla","NeutroReso"]
col_MSCs = ["MSCs","MSCinfla","MSCreso"]
col_macrocircs = ["MacroCirc","MacroCircInfla","MacroCircReso"]
col_macrores = ["MacroRes","MacroResInfla","MacroResReso"]
col_deads = ["Apoptotic","Necrotic"]
local_cells = ["MSCs","MSCinfla","MSCreso","MacroRes","MacroResInfla","MacroResReso","BloodVessels","OtherCells"]

t0=time.time()

print("start")

d_evals = []
d_tests = []

count = 0

path = "batchs_300k"

suff="_300k"

cols = ['step',
'APOP', 'DANGER', 'INFLA', 'RESO',
'Apoptotic', 'Necrotic',
'BloodVessels', 'OtherCells',
'MSCinfla', 'MSCreso', 'MSCs',
'MacroCirc', 'MacroCircInfla', 'MacroCircReso',
'MacroRes', 'MacroResInfla', 'MacroResReso',
'NeutroInfla', 'NeutroReso', 'Neutrophiles']

if os.path.isdir(path):
    shutil.rmtree(path)
os.makedirs(path, exist_ok=True)

with DatabaseManager("LHS_07_25.db") as db:

    for (id_param,id_replicat),df in db.iterOn('data', group=('ID_PARAMETER','ID_REPLICAT')):
        nb_evals = len(d_evals)
        if not nb_evals%1000: print(count*100_000 + nb_evals)
        if not nb_evals%10_000: 
            print(f"timing ({count*100_000 + nb_evals}) = {((time.time()-t0)/3600):.2f}h")
            ram = psutil.virtual_memory()
            print("RAM usage (%):", ram.percent)

        df = df.rename(columns={"index":"step"})

        df = df[cols]

        loss_ratio_thd = 0.5

        stromal_loss_thd = df[["MSCinfla","MSCreso","MSCs"]].sum(axis=1).iloc[10] * loss_ratio_thd
        macrores_loss_thd = df[["MacroRes","MacroResInfla","MacroResReso"]].sum(axis=1).iloc[10] * loss_ratio_thd
        local_cells_loss_thd = df[["MacroRes","MacroResInfla","MacroResReso",
                                "MSCinfla","MSCreso","MSCs","OtherCells","BloodVessels"]].sum(axis=1).iloc[10] * loss_ratio_thd
        
        d_evals.append({
            "ID_PARAMETER": id_param,
            "ID_REPLICAT": id_replicat,
            "total_macrocircs": total_macrocircs_eval(df, col_macrocircs),
            "total_neutros": total_neutros_eval(df, col_neutros),
            "macrocircs_timing": macrocircs_timing_eval(df, col_macrocircs),
            "neutros_timing": neutros_timing_eval(df, col_neutros),
            "neutro_at_end": neutro_at_end_eval(df, col_neutros),
            "apop_at_end": apop_at_end_eval(df),
            "necro_at_end": necro_at_end_eval(df),
            "stromal_loss": stromal_loss_eval(df, col_MSCs),
            "macrores_loss": macrores_loss_eval(df, col_macrores),
            "local_cells_loss": local_cells_loss_eval(df, local_cells),
            "infla_timing": infla_timing_eval(df),
            "reso_timing": reso_timing_eval(df),
            "infla_at_end": infla_at_end_eval(df)})

        d_thd_data = {
            "total_macrocircs_thd": 0, 
            "total_neutros_thd": 0,
            "macrocircs_neutros_delta_thd": 0,
            "neutros_timing_range": (60,240), 
            "neutro_at_end_thd": 0,
            "apop_at_end_thd": 0,
            "necro_at_end_thd": 0, 
            "stromal_loss_thd": stromal_loss_thd,
            "macrores_loss_thd": macrores_loss_thd,
            "local_cells_loss_thd": local_cells_loss_thd,
            "infla_timing_range": (0,480),
            "reso_infla_delta_timing_thd": 0,
            "infla_at_end_thd": 1,}

        last_eval = d_evals[-1]

        
        d_tests.append({
            "ID_PARAMETER": id_param,
            "ID_REPLICAT": id_replicat,
            "circulant_recruited": circulant_recruited_test(last_eval["total_macrocircs"], last_eval["total_neutros"], d_thd_data["total_macrocircs_thd"], d_thd_data["total_neutros_thd"]),
            "circulant_sequenced": circulant_sequenced_test(last_eval["macrocircs_timing"], last_eval["neutros_timing"], d_thd_data["macrocircs_neutros_delta_thd"]),
            "neutros_peak_timing": neutros_peak_timing_test(last_eval["neutros_timing"], d_thd_data["neutros_timing_range"]), 
            "neutro_cleared": neutro_cleared_test(last_eval["neutro_at_end"], d_thd_data["neutro_at_end_thd"]),
            "apop_cleared": apop_cleared_test(last_eval["apop_at_end"], d_thd_data["apop_at_end_thd"]),
            "necro_cleared": necro_cleared_test(last_eval["necro_at_end"], d_thd_data["necro_at_end_thd"]), 
            "stromal_spared": stromal_spared_test(last_eval["stromal_loss"], d_thd_data["stromal_loss_thd"]),
            "macrores_spared": macrores_spared_test(last_eval["macrores_loss"], d_thd_data["macrores_loss_thd"]),
            "tissue_spared": tissue_spared_test(last_eval["local_cells_loss"], d_thd_data["local_cells_loss_thd"]),
            "infla_timed": infla_timed_test(last_eval["infla_timing"], d_thd_data["infla_timing_range"]),
            "reso_infla_sequenced": reso_infla_sequenced_test(last_eval["infla_timing"], last_eval["reso_timing"], d_thd_data["reso_infla_delta_timing_thd"]),
            "infla_cleared": infla_cleared_test(last_eval["infla_at_end"], d_thd_data["infla_at_end_thd"]),})
        
        nb_evals = len(d_evals)
        if nb_evals > 0 and not nb_evals%300_000:
            pd.DataFrame(d_evals).to_feather(f"{path}/df_evals{suff}_{count}.fth")
            pd.DataFrame(d_tests).to_feather(f"{path}/df_tests{suff}_{count}.fth")

            d_evals.clear()
            d_tests.clear()
            count+=1

        del(df)
        gc.collect()

    if len(d_evals) > 0 and len(d_tests) > 0:
        pd.DataFrame(d_evals).to_feather(f"{path}/df_evals{suff}_{count}.fth")
        pd.DataFrame(d_tests).to_feather(f"{path}/df_tests{suff}_{count}.fth")



print(f"finished t={((time.time()-t0)/3600):.2f}h")

