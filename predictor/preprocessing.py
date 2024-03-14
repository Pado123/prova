from collections import Counter
from datetime import datetime
import pandas as pd
import os
import numpy as np
import pm4py
import pickle
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def moving_average(data, window_size):
    if window_size <= 0 or window_size > len(data):
        raise ValueError("Invalid window size")
    data = np.array(data)
    moving_averages = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return moving_averages

def closest_index(L, n):
    
    # Ensure that the list is not empty
    if not L:
        return None

    # Calculate the absolute difference between each element and n
    differences = [abs(x - n) for x in L]

    # Find the index of the minimum difference
    closest_index = int(differences.index(min(differences)))

    return closest_index

def add_exp_encoding(el, hparams, rlt):    
                
        resources_dist = rlt[el[hparams["res_"]]][el[hparams["act_"]]]  
        resources_dist = sorted(resources_dist, key=lambda resources_dist: resources_dist[1]) 
        exp_vec_len = len(resources_dist)
        dates = [int(x[1]) for x in resources_dist]
        times = [int(x[0]) for x in resources_dist]
    
        if len(times) < 2:
            raise Exception("Not enough resources")
            # return exp_vec_len, dates[0], dates[0], dates[0], dates[0]
        
        #Just consider traces after the current time of the activity
        if '17' in hparams["exp_name"]:
            breakp = closest_index(dates, el['start_times'])
        else:
            breakp = closest_index(dates, el[hparams["end_"]])
        if breakp in {0,1}:
            raise Exception("Breakp too early")
            # return exp_vec_len, dates[0], dates[0], dates[-1], dates[-1]
            
        if len(dates) in {2, 3, 4}:
            raise Exception("Not enough resources")
            # return exp_vec_len, dates[0], dates[0], dates[-1], dates[-1]
        
        one_day, five_days, twenty_days = dates[breakp] + 86400, dates[breakp] + 5*86400, dates[breakp] + 20*86400 
        idx_1, idx_5, idx_20 = closest_index(dates, one_day), closest_index(dates, five_days), closest_index(dates, twenty_days)
        time_after_1, time_after_5, time_after_20 = int(np.mean(times[idx_1-2 : idx_1+2])), int(np.mean(times[idx_5-2 : idx_5+2])), int(np.mean(times[idx_20-2 : idx_20+2]))
        
        # print(time_after_1, time_after_5, time_after_20)
        
        dates = dates[:breakp]
        times = times[:breakp]
        
        t0, t100 = times[0], times[-1]
        d0, d100 = np.mean(dates[0:+5]), np.mean(dates[-5:-1])
        d33, d66 = d0 + (d100-d0)*.33, d0 + (d100-d0)*.66
        d33, d66 = closest_index(dates, d33), closest_index(dates, d66)
        t33, t66 = times[d33], times[d66]
        return exp_vec_len, t0, t33, t66, t100, time_after_1, time_after_5, time_after_20
                
def preprocess_log(log, hparams):
    
    pp_log = pd.DataFrame(columns=(list(log.columns[:-1])+["exp_vec_len", "t0", "t1", "t2", "t3"]+["time_after_1", "time_after_5", "time_after_20"]+[log.columns[-1]])) #, "d01", "d12", "d23"])
    rlt = pickle.load(open(f"variables/{hparams['exp_name']}/res_learning_times.pkl", 'rb'))
    missing_values=0
    
    for line in tqdm.tqdm(log.index):
        el = log.loc[line]
        
        try:
            exp_vec_len, d0, d33, d66, d100, time_after_1, time_after_5, time_after_20 = add_exp_encoding(el, hparams, rlt)
            pp_log.loc[line] = list(el.values[:-1])+[exp_vec_len, d0, d33, d66, d100]+[time_after_1, time_after_5, time_after_20]+[el.values[-1]]
            # print('lets go')
        except: 
            missing_values+=1
            print("Missone")

    print(f"Used the percentage of {100*(1-missing_values/len(log))}% of the log")
    return pp_log

def create_resources_times(log, split_time, hparams):
    
    # Check if there is already a dictionary saved 
    if os.path.exists(f"variables/{hparams['exp_name']}/resource_availability_dict.pkl"):
        return pickle.load(open(f"variables/{hparams['exp_name']}/resource_availability_dict.pkl", 'rb'))    
    
    resources_times = {}
    if 'BAC' in hparams["exp_name"]:
        for res in tqdm.tqdm(log[hparams["res_"]].unique()):
            rlog = log[log[hparams["res_"]] == res]
            
            # Filter the log with start less or equal than split_time
            rlog = rlog[rlog[hparams["start_"]] <= split_time]
            
            print(len(rlog[rlog[hparams["end_"]] == rlog[hparams["end_"]].max()]))
            
            # return the maximum end time 
            resources_times[res] = rlog[hparams["end_"]].max()
            
    elif '17' in hparams["exp_name"]:
        for res in tqdm.tqdm(log[hparams["res_"]].unique()):
            rlog = log[log[hparams["res_"]] == res]
            
            # Filter the log with start less or equal than split_time
            resources_times[res] = rlog[rlog['start_times'] <= split_time]['start_times'].max()
    pickle.dump(resources_times, open(f"variables/{hparams['exp_name']}/resource_availability_dict.pkl", 'wb'))
    return resources_times


# %%


# %%
