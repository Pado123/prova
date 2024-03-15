# %%
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

# Ignor FutureWarnings
warnings.filterwarnings('ignore')

# Set the random seed
random.seed(1618)

# read the parameter json dictionary
hparams = json.load(open('hparams/hparams_bac.json'))
print('Exp name is ', hparams['exp_name'])

# Create a folder called "plots" or "results" if not present
if not os.path.exists('plots'): os.mkdir('plots')
if not os.path.exists('results'): os.mkdir('results')

# Create a folder with the experiment name if not present
if not os.path.exists(f"plots/{hparams['exp_name']}"): os.mkdir(f"plots/{hparams['exp_name']}")
if not os.path.exists(f"results/{hparams['exp_name']}"): os.mkdir(f"results/{hparams['exp_name']}")

# Create a folder plotvars into plots folder if not present 
if not os.path.exists(f"plots/{hparams['exp_name']}/plotvars"): os.mkdir(f"plots/{hparams['exp_name']}/plotvars")

import utils
import IO
import reading_log
import plotting


# %% Is not existing, create results folder and subfolders with the experiment_name 
if not os.path.exists('results'): os.mkdir('results')
if not os.path.exists(f"results/{hparams['exp_name']}"): os.mkdir(f"results/{hparams['exp_name']}")

# read the log
log = reading_log.read_log(hparams)
if '17' in hparams['exp_name']:
    log = utils.convert_timestamps_to_unix(log, hparams)
    print('Timestamps converted to unix')
    
log = pd.read_csv(log)

# Filter the log only keeping activities that depecnds on bank's staff
log = utils.filter_on_banking_activities(log, hparams['banking_activities'])

# # timestamps to unix
log = utils.convert_timestamps_to_unix(log, hparams)

# plotting.plot_traces_lenght_distribution(log, hparams)
    
# plotting.plot_active_traces_dist(log, hparams)    

# utils.create_trace_len_dict(log, hparams)
plotting.plot_lenght_distribution(log, hparams)

# %%
lifecycles = 'lifecycle' in hparams.keys()
utils.create_activity_duration_dict(log, hparams, lifecycles=lifecycles)


# %% Load it and reorganize it for general plots
drifts = [1470100000, 1475100000]
if hparams["drift_"]=='1':
    plotting.plot_activity_durations_with_drift(drift=drifts, hparams=hparams, w_den=6)
if hparams["drift_"]=='0':
    plotting.plot_activity_durations_without_drift(hparams=hparams, w_den=6)
    
    

# %% Read preprocessed log
train_idxs, test_idxs = utils.get_split_indexes(log, hparams, train_size=0.7)


# %%
import predictor.preprocessing as ppg
#Preprocessing imported from other folder
log_preprocessed = pd.read_csv(hparams["preprocessed_log_path"])
resources_dict = utils.get_resources_times(log_preprocessed, hparams)

# Reorder the log for having the case column as first column
log_preprocessed = log_preprocessed[[hparams["case_"]] + [col for col in log_preprocessed.columns if col != hparams["case_"]]]

train_indexes = pickle.load(open(f"variables/{hparams['exp_name']}/train_idx.pkl", "rb")) 
test_indexes = pickle.load(open(f"variables/{hparams['exp_name']}/test_idx.pkl", "rb"))
dfTrain = log_preprocessed[log_preprocessed[hparams["case_"]].isin(train_indexes)] # TODO: I test idx son quelli dello split con max 
dfTest = log_preprocessed[~log_preprocessed[hparams["case_"]].isin(train_indexes)]
print('Train and test indexes loaded')
print('They have shape of', dfTrain.shape, dfTest.shape)


# %% 
train_preprocessed = ppg.preprocess_log(dfTrain, hparams) 
train_preprocessed.to_csv(f"logs/{hparams['exp_name']}_dfTrain_processed_for_train.csv", index=False)
test_preprocessed = ppg.preprocess_log(dfTest, hparams)
test_preprocessed.to_csv(f"logs/{hparams['exp_name']}_dfTest_processed_for_test.csv", index=False)
print('Preprocessed logs saved')
print('They have shape of', train_preprocessed.shape, test_preprocessed.shape)

# %% Trainer for models
import predictor.time_pred as tp
train_preprocessed = pd.read_csv(f"logs/{hparams['exp_name']}_dfTrain_processed_for_train.csv")
model1 = tp.preprocess_and_train(train_preprocessed, hparams, model_name="model1")
IO.write(model1, f"results/{hparams['exp_name']}/model1.pkl")
model5 = tp.preprocess_and_train(train_preprocessed, hparams, model_name="model5")
IO.write(model5, f"results/{hparams['exp_name']}/model5.pkl")
model20 = tp.preprocess_and_train(train_preprocessed, hparams, model_name="model20")
IO.write(model20, f"results/{hparams['exp_name']}/model20.pkl")
model_time = tp.preprocess_and_train(train_preprocessed, hparams, model_name="time")
IO.write(model_time, f"results/{hparams['exp_name']}/model_time.pkl")

# %% 
import hash_maps
import predictor.preprocessing as ppg
# import the test set
train_preprocessed = pd.read_csv(f"logs/{hparams['exp_name']}_dfTrain_processed_for_train.csv") 
test_preprocessed = pd.read_csv(f"logs/{hparams['exp_name']}_dfTest_processed_for_test.csv")
split_time = test_preprocessed.END_DATE.mean() - 198567 # For BAC
# split_time = 1464549028 # For BPI17 before
# split_time = 1480703513 # For BPI17 after
res_availability_dict = ppg.create_resources_times(log_preprocessed, split_time, hparams)
running_log = utils.create_running_log(test_preprocessed, hparams, split_time)
running_log.to_csv(f"logs/{hparams['exp_name']}_running_log.csv")
running_log = pd.read_csv(f"logs/{hparams['exp_name']}_running_log.csv", index_col=0)

dfTrain = log_preprocessed[log_preprocessed[hparams["case_"]].isin(train_indexes)]
activities_discovery = hash_maps.discover_activities(dfTrain, hparams)
pickle.dump(activities_discovery, open(f"variables/{hparams['exp_name']}/activities_discovery.pkl", "wb"))

# %% 
import predictor.time_pred as tp
# train_preprocessed = pd.read_csv(f"logs/{hparams['exp_name']}_dfTrain.csv") 
transitions_system = hash_maps.create_transition_system(dfTrain, hparams, node_level=False, thrs=.02) 
pickle.dump(transitions_system, open(f"variables/{hparams['exp_name']}/transitions_system.pkl", "wb"))

running_log = pd.read_csv(f"logs/{hparams['exp_name']}_running_log.csv", index_col=0)
rank_indexes = utils.generate_rank_indexes(running_log, hparams) 

train_indexes = pickle.load(open(f"variables/{hparams['exp_name']}/train_idx.pkl", "rb"))
log_preprocessed = pd.read_csv(hparams["preprocessed_log_path"])
rank_indexes = pickle.load(open(f"results/{hparams['exp_name']}/rank_indexes.pkl", "rb"))

running_log = pd.read_csv(f"logs/{hparams['exp_name']}_running_log.csv", index_col=0)
activities_discovery = pickle.load(open(f"variables/{hparams['exp_name']}/activities_discovery.pkl", "rb"))

# Different parts of the oracle function
model1 = IO.read(f"results/{hparams['exp_name']}/model1.pkl") #Lambda_1
model5 = IO.read(f"results/{hparams['exp_name']}/model5.pkl") #Lambda_5
model20 = IO.read(f"results/{hparams['exp_name']}/model20.pkl") #Lambda_20
model_time = IO.read(f"results/{hparams['exp_name']}/model_time.pkl") #Rem_time_function



# %% Generate First Profile
skipped_traces = 0
available_resources = dfTest[hparams['res_']].unique()
busy_resources = [] 
benchmark = False # This hparam set if it is or not the BENCHMARK MODE, as referred in the paper.
print('Start generating recommendations', not(benchmark)*'not', 'considering the benchmark')


if not benchmark:
    n = 3 # Number of res per act
    k = 25 # Number of recommendations pairs in total (at max n*max(n_acts_possible))
        
    recommendations_dataframe = pd.DataFrame(columns=["case:concept:name", "repl_id"]+[f'act_{i}' for i in range(1, k+1)]+[f'res_{i}' for i in range(1, k+1)])
    for trace_id in tqdm.tqdm([el[0] for el in rank_indexes]):
        
        # This traces are already ordered by predicted total time in a decrescent order
        # Filter the trace from the running log
        trace = running_log[running_log[hparams["case_"]] == trace_id]
        last_event = trace.iloc[-1]
        # print(f'Last activity is {last_event[hparams["act_"]]}')
        
        # Find it in transition systems
        repl_id, activity_list = hash_maps.get_repl_id_and_acts(last_event, dfTest, hparams) 
        preprocessed_for_pred_trace = test_preprocessed[test_preprocessed[hparams["case_"]] == trace_id]
        
        #Find next activity from transition system
        try:
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
            # From the column with the lowest expected value_time, return activity and resource
            rec_df.sort_values(by='expected_value_time', inplace=True)
            
            # Drop the rows in which the resource is missing
            rec_df = rec_df[rec_df['res'] != 'missing']
            
            # Filter the rec_df on the top n activities
            rec_df = utils.filter_on_topk_acts(rec_df, n)
            rec_df = rec_df.iloc[:k].reset_index(drop=True)
            
            
            # From the rec_df dataframe, append it to the recommendations_dataframe following the order
            vec = [trace_id, repl_id]         
            if len(rec_df) >= k:
                vec = vec + [rec_df.iloc[i]['act'] for i in range(k)]
                vec = vec + [rec_df.iloc[i]['res'] for i in range(k)]
                recommendations_dataframe = recommendations_dataframe.append(pd.Series(vec, index=recommendations_dataframe.columns), ignore_index=True)
                print(f'The recommendations dataframe is {len(recommendations_dataframe)}')
            else:
                vec = vec + [rec_df.iloc[i]['act'] for i in range(len(rec_df))]
                vec = vec + ['missing' for i in range(k-len(rec_df))]
                vec = vec + [rec_df.iloc[i]['res'] for i in range(len(rec_df))]
                vec = vec + ['missing' for i in range(k-len(rec_df))]
                recommendations_dataframe = recommendations_dataframe.append(pd.Series(vec, index=recommendations_dataframe.columns), ignore_index=True)
                print(f'The recommendations dataframe is {len(recommendations_dataframe)}')
                
            if len(recommendations_dataframe) == 10:
                recommendations_dataframe.to_csv(f"results/{hparams['exp_name']}/recommendations_dataframe_10.csv", index=False)
            if len(recommendations_dataframe) == 50:
                recommendations_dataframe.to_csv(f"results/{hparams['exp_name']}/recommendations_dataframe_50.csv", index=False)
        except:
            skipped_traces += 1
            print(f"The trace is not in the transition system, skipped traces are {skipped_traces}")
            continue
        
    recommendations_dataframe.to_csv(f"results/{hparams['exp_name']}/recommendations_dataframe.csv", index=False)
    print(f"Skipped traces are {skipped_traces}")
    print(f"Total traces are {len(rank_indexes)}")
    
elif benchmark:
    recommendations_dataframe = pd.DataFrame(columns=["case:concept:name", "repl_id", "act_1", "res_1"])
    for trace_id in tqdm.tqdm([el[0] for el in rank_indexes]):
        
        # This traces are already ordered by predicted total time in a decrescent order
        # Filter the trace from the running log
        trace = running_log[running_log[hparams["case_"]] == trace_id]
        last_event = trace.iloc[-1]
        repl_id, activity_list = hash_maps.get_repl_id_and_acts(last_event, dfTest, hparams) 
        preprocessed_for_pred_trace = test_preprocessed[test_preprocessed[hparams["case_"]] == trace_id]
        
        #Find next activity from transition system
        try:
            rec_df = pd.DataFrame(columns=["act", "res"])
            next_possible_acts = transitions_system[''.join(activity_list)]
            # print(f'Next possible acts are {next_possible_acts}')
            for act in next_possible_acts:
                if act != activity_list[-1]:
                    for res in activities_discovery[act]:
                        rec_df = rec_df.append({"act": act, "res": res}, ignore_index=True)
            
            # rec_df['expected_value_time1'] = [tp.apply_pred(last_event, [rec_df.iloc[i].values], model1, hparams) for i in range(len(rec_df))]
            # rec_df['expected_value_time5'] = [tp.apply_pred(last_event, [rec_df.iloc[i].values], model5, hparams) for i in range(len(rec_df))]
            # rec_df['expected_value_time20'] = [tp.apply_pred(last_event, [rec_df.iloc[i].values], model20, hparams) for i in range(len(rec_df))]
            rec_df['expected_value_time'] = [tp.apply_pred(last_event, [rec_df.iloc[i].values], model_time, hparams) for i in range(len(rec_df))]
            
            # From the column with the lowest expected value_time, return activity and resource
            rec_df.sort_values(by='expected_value_time', inplace=True)
            
            # # Remove the rows with the first activity
            # rec_df = rec_df[rec_df['act'] != rec_df['act'][0]]
                        
            # Drop the rows in which the resource is missing
            rec_df = rec_df[rec_df['res'] != 'missing']
            if len(busy_resources) < 45 : #len(available_resources):
                rec_df = rec_df[~rec_df['res'].isin(busy_resources)]                
                vec = [trace_id, repl_id, rec_df.iloc[0]['act'], rec_df.iloc[0]['res']]
                recommendations_dataframe = recommendations_dataframe.append(pd.Series(vec, index=recommendations_dataframe.columns), ignore_index=True)
                busy_resources.append(rec_df.iloc[0]['res'])
                print(f'Busy resources are {len(busy_resources)}, and {rec_df.iloc[0]["res"]} is busy')
            else:
                vec = [trace_id, repl_id, rec_df.iloc[0]['act'], rec_df.iloc[0]['res']]
                recommendations_dataframe = recommendations_dataframe.append(pd.Series(vec, index=recommendations_dataframe.columns), ignore_index=True)

            print(f'The recommendations dataframe is {len(recommendations_dataframe)}')
            
        except:
            skipped_traces += 1
            print("The trace is not in the transition system")
            continue
        if len(recommendations_dataframe) == 150:
            recommendations_dataframe['res_1'] = recommendations_dataframe['res_1'].astype(str)
            recommendations_dataframe.to_csv(f"results/{hparams['exp_name']}/recommendations_dataframe_150_selfish.csv", index=False)
                
    recommendations_dataframe.to_csv(f"results/{hparams['exp_name']}/recommendations_dataframe_selfish.csv", index=False)
    print(f"Skipped traces are {recommendations_dataframe}")
    print(f"Total traces are {len(rank_indexes)}")


# %%
