import pandas as pd
import tqdm 
import pickle
import os 

def create_transition_system(log, hparams, node_level=False, thrs=.02):

    # Check if there is a transition system already created
    if f'transitions_system.pkl' in os.listdir(f'variables/{hparams["exp_name"]}'):
        print('Transition system already created, read it from the variables folder')
        return pickle.load(open(f'variables/{hparams["exp_name"]}/transitions_system.pkl', 'rb'))
    
    print('Creating transition system')
    if not node_level:        
        if '17' in hparams['exp_name']:
            transition_system = dict()
            
            for index_trace in tqdm.tqdm(log[hparams['case_']].unique()[1:]):
                
                trace = log[log[hparams['case_']] == index_trace].reset_index(drop=True)
                activity_list = trace[hparams['act_']].tolist()
                
                for i in range(1, len(activity_list) - 1):
                    
                    if ''.join(activity_list[:i]) not in transition_system.keys():
                        transition_system[''.join(activity_list[:i])] = [activity_list[i+1]]
                    else:
                        if activity_list[i+1] not in transition_system[''.join(activity_list[:i])]:
                            transition_system[''.join(activity_list[:i])].append(activity_list[i+1])

        else:
            transition_system = dict()
            
            for index_trace in tqdm.tqdm(log[hparams['case_']].unique()[1:]):
                
                trace = log[log[hparams['case_']] == index_trace].reset_index(drop=True)
                activity_list = trace[hparams['act_']].tolist()
                
                for i in range(1, len(activity_list) - 1):
                    if ''.join(activity_list[:i]) not in transition_system.keys():
                        transition_system[''.join(activity_list[:i])] = dict()
                        transition_system[''.join(activity_list[:i])][activity_list[i+1]] = 1
                    else:
                        if activity_list[i+1] not in transition_system[''.join(activity_list[:i])].keys():
                            transition_system[''.join(activity_list[:i])][activity_list[i+1]] = 1
                        else:
                            transition_system[''.join(activity_list[:i])][activity_list[i+1]] += 1
                            
            # sum all the values in the dictionary
            thrs = 0.0005
            total = 0 
            drop_list = []
            ret_dict = dict()
            for key in transition_system:
                for key2 in transition_system[key]:
                    total += transition_system[key][key2]
                    
            for key in transition_system.keys():
                ret_dict[key] = []
                for key2 in transition_system[key]:
                    transition_system[key][key2] /= total
                    if transition_system[key][key2] > thrs:
                        ret_dict[key].append(key2)
                
        print('Transition system created')
        return ret_dict

        
def get_repl_id_and_acts(last_event, dfTest, hparams):
    
    # Filter the dfTEst on the trace id of last event
    trace = dfTest[dfTest[hparams["case_"]] == last_event[hparams["case_"]]].reset_index(drop=True)
    
    # Get the index of the last event using the end timestamp
    if '17' in hparams["exp_name"]:
        repl_id = trace[trace['start_times'] == last_event["start_times"]].index[0]
    elif 'BAC' in hparams["exp_name"]:
        repl_id = trace[trace[hparams["end_"]] == last_event[hparams["end_"]]].index[0]
    
    return repl_id, list(trace[hparams["act_"]])[:repl_id]

def discover_activities(dfTrain, hparams):
    
    # Check if there is already a dictionary saved
    if os.path.exists(f"variables/{hparams['exp_name']}/activities_discovery.pkl"):
        print('Activities already discovered, read it from the variables folder')
        return pickle.load(open(f"variables/{hparams['exp_name']}/activities_discovery.pkl", 'rb'))   
    
    # Create a dictionary with the activities as keys
    activities_discovery = dict()
    
    for idx in tqdm.tqdm(dfTrain.index):
        
        line = dfTrain.loc[idx]
        
        if line[hparams['act_']] not in activities_discovery.keys():
            activities_discovery[line[hparams['act_']]] = [line[hparams['res_']]]

        else:
            if line[hparams['res_']] not in activities_discovery[line[hparams['act_']]]:
                activities_discovery[line[hparams['act_']]].append(line[hparams['res_']])
    
    return activities_discovery
    
