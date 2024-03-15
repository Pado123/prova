import pandas as pd 
import pm4py 
import sklearn
import json
import numpy as np
import os
import random
import tqdm
import pickle
import matplotlib.pyplot as plt
import tqdm 

import warnings

import IO

hparams = json.load(open('hparams/hparams_bac.json'))
first_profile = pd.read_csv(f"profiles/{hparams['exp_name']}/first_profile.csv", index=False)

# Import Different parts of the oracle function
model1 = IO.read(f"results/{hparams['exp_name']}/model1.pkl") #Lambda_1
model5 = IO.read(f"results/{hparams['exp_name']}/model5.pkl") #Lambda_5
model20 = IO.read(f"results/{hparams['exp_name']}/model20.pkl") #Lambda_20
model_time = IO.read(f"results/{hparams['exp_name']}/model_time.pkl") #Rem_time_function
running_log  = IO.read(f"results/{hparams['exp_name']}/running_log.pkl") #Running log


def evaluate_Time_Workload_Coefficient(profile):
    
    first_term = np.sum(profile['expected_value_time'])/198567 #for BAC, 1464549028 For BPI17 before, 1480703513 # For BPI17 after
    
    second_term = (np.std(profile['expected_value_time1']) + np.std(profile['expected_value_time5']) + np.std(profile['expected_value_time20']))/(3*198567) #for BAC, 1464549028 For BPI17 before, 1480703513 # For BPI17 after

    third_term = 1-len(profile['res'].unique())/640 #for BAC, 105 For BPI17 before, 105 # For BPI17 after
    
    return (first_term + second_term + third_term)/3

def modify_profile(profile):
    
    #Pick a random element in 1,len(profile)
    index = random.randint(1,len(profile)-1)
    
    trace_identifier = profile.loc[index]['case:concept:name']  
    profile = profile.drop(index)
    
    #Get the possible next activities and res
    # This traces are already ordered by predicted total time in a decrescent order
    # Filter the trace from the running log
    trace = running_log[running_log[hparams["case_"]] == trace_identifier]
    last_event = trace.iloc[-1]
    # print(f'Last activity is {last_event[hparams["act_"]]}')
    
    # Find it in transition systems
    repl_id, activity_list = hash_maps.get_repl_id_and_acts(last_event, dfTest, hparams) 
    preprocessed_for_pred_trace = test_preprocessed[test_preprocessed[hparams["case_"]] == trace_identifier]
    
    #Find next activity from transition system
    rec_df = pd.DataFrame(columns=["act", "res"])
    next_possible_acts = transitions_system[''.join(activity_list)]
    # print(f'Next possible acts are {next_possible_acts}')
    for act in next_possible_acts:
        if act != activity_list[-1]:
            for res in activities_discovery[act]:
                rec_df = rec_df.append({"act": act, "res": res}, ignore_index=True)
    
    rec_df['expected_value_time1'] = [tp.apply_pred(last_event, [rec_df.iloc[i].values], model1, hparams) for i in range(len(rec_df))]
    rec_df['expected_value_time5'] = [tp.apply_pred(last_event, [rec_df.iloc[i].values], model5, hparams) for i in range(len(rec_df))]
    rec_df['expected_value_time20'] = [tp.apply_pred(last_event, [rec_df.iloc[i].values], model20, hparams) for i in range(len(rec_df))]
    rec_df['expected_value_time'] = [tp.apply_pred(last_event, [rec_df.iloc[i].values], model_time, hparams) for i in range(len(rec_df))]
    
    index = random.randint(1,len(rec_df)-1)
    profile = profile.append(rec_df.loc[index], ignore_index=True)


ranked_profiles = [(first_profile, evaluate_Time_Workload_Coefficient(first_profile))]
early_stop_param = 0
old_profile = first_profile

while early_stop_param < 200:
    
    old_score = evaluate_Time_Workload_Coefficient(old_profile)
    new_profile = modify_profile(first_profile)
    new_score = evaluate_Time_Workload_Coefficient(new_profile)
    ranked_profiles.append((new_profile, new_score))
    
    if new_score < old_score:
        early_stop_param = 0 
        old_profile = new_profile
    else:
        early_stop_param += 1
    if len(ranked_profiles) % 100 == 0:
        pickle.dump(ranked_profiles, open(f"profiles/{hparams['exp_name']}/ranked_profiles.pkl", "wb"))
        
id_list = [i for i in running_log[hparams["case_"]].unique()]
def provide_recs(id_list):
    ret_dict = {}
    k = 3
    while k < 3:
        for i in id_list:
            if (ranked_profiles[i][0]['case:concept:name']['act'], ranked_profiles[i][0]['case:concept:name']['res']) not in id_list:
                if ret_dict[i] not in ret_dict.keys() :
                    ret_dict[i] = [(ranked_profiles[i][0]['case:concept:name']['act'], ranked_profiles[i][0]['case:concept:name']['res'])]
                else:
                    ret_dict[i].append((ranked_profiles[i][0]['case:concept:name']['act'], ranked_profiles[i][0]['case:concept:name']['res']))
        k += 1
    pickle.dump(ret_dict, open(f"profiles/{hparams['exp_name']}/recommendations.pkl", "wb"))