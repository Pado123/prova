import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import pickle
import os
sns.set_style('darkgrid')

from IO import read, write

def get_split_indexes(df, hparams, train_size=.70):
    
    # df has replaced log cause am lazy, but it is just in that function
    
    try :
        case_id_name, start_date_name = hparams["case_"], hparams["start_"]  
    except: 
        case_id_name, start_date_name = hparams["case_"], hparams["timestamp_"]  
    
    print('Starting splitting procedure with train_size = ', train_size)
    start_end_couple = list()
    if start_date_name=='timestamp_':
        for idx in tqdm.tqdm(df[case_id_name].unique()):
            df_ = df[df[case_id_name] == idx].reset_index(drop=True)
            start_end_couple.append([idx, df_[start_date_name].values[0], df_[start_date_name].values[len(df_) - 1]])
        start_end_couple = pd.DataFrame(start_end_couple, columns=['idx', 'start', 'end'])
    
    else:
        for idx in tqdm.tqdm(df[case_id_name].unique()):
            df_ = df[df[case_id_name] == idx].reset_index(drop=True)
            start_end_couple.append([idx, df_[start_date_name].values[0], df_['END_DATE'].values[len(df_) - 1]])
        start_end_couple = pd.DataFrame(start_end_couple, columns=['idx', 'start', 'end'])
        
        
    # if the type of the start and end column are not unix, convert them
    if '17' in hparams['exp_name']:
        start_end_couple['end'] = [int(pd.Timestamp(i).timestamp()) for i in start_end_couple['end'].values] 
        start_end_couple['start'] = [int(pd.Timestamp(i).timestamp()) for i in start_end_couple['start'].values]
    else:
        None
        
    print(f'The min max range is {start_end_couple.start.min()}, {start_end_couple.end.max()}')
    print(f'With length {start_end_couple.end.max() - start_end_couple.start.min()}')

    # Initialize pdf of active cases and cdf of closed cases
    times_dict_pdf = dict()
    times_dict_cdf = dict()
    split = int(
        (start_end_couple.end.max() - start_end_couple.start.min()) / 10000)  # In order to get a 10000 dotted graph
    for time in range(int(start_end_couple.start.min()), int(start_end_couple.end.max()), split):
        times_dict_pdf[time] = 0
        times_dict_cdf[time] = 0

    for time in tqdm.tqdm(range(int(start_end_couple.start.min()), int(start_end_couple.end.max()), split)):
        for line in np.array(start_end_couple[['start', 'end']]):
            line = np.array(line)
            if (line[0] <= time) and (line[1] >= time):
                times_dict_pdf[time] += 1
    for time in tqdm.tqdm(range(int(start_end_couple.start.min()), int(start_end_couple.end.max()), split)):
        for line in np.array(start_end_couple[['start', 'end']]):
            line = np.array(line)
            if (line[1] <= time):  # Keep just k closes cases
                times_dict_cdf[time] += 1

    sns.set_style('darkgrid')
    plt.title('Number of active operations')
    plt.xlabel('Time')
    plt.ylabel('Count')

    sns.lineplot([times_dict_pdf.keys(), times_dict_pdf.values()])
    sns.lineplot([times_dict_cdf.keys(), times_dict_cdf.values()])
    plt.savefig('Active and completed cases distribution.png')
    times_dist = pd.DataFrame(columns=['times', 'pdf_active', 'cdf_closed'])
    times_dist['times'] = times_dict_pdf.keys()
    times_dist['pdf_active'] = times_dict_pdf.values()
    times_dist['cdf_closed'] = np.array(list(times_dict_cdf.values())) / (len(df[case_id_name].unique()))
    # Set threshold after 60 of closed activities (it'll be the train set)

    test_dim = times_dist[times_dist.cdf_closed > train_size].pdf_active.max()
    thrs = times_dist[times_dist.pdf_active == test_dim].times.values[0]
    train_idxs = start_end_couple[start_end_couple['end'] <= thrs]['idx'].values
    test_idxs = start_end_couple[start_end_couple['end'] >= thrs][start_end_couple['start'] <= thrs]['idx'].values

    pickle.dump(train_idxs, open(f'variables/train_idx_{hparams["exp_name"]}.pkl', 'wb'))
    pickle.dump(test_idxs, open(f'variables/test_idx_{hparams["exp_name"]}.pkl', 'wb'))
    return train_idxs, test_idxs

def create_running_log(test_preprocessed, hparams, split_time):

    # Check if there is a running log in the log folder
    if f"{hparams['exp_name']}_running_log.csv" in os.listdir(f"logs"):
        print('Running log already created, read it from the variables folder')
        return pd.read_csv(f"logs/{hparams['exp_name']}_running_log.csv", index_col=0)
    
    running_log = pd.DataFrame(columns = test_preprocessed.columns)
    
    for idx in tqdm.tqdm(test_preprocessed[hparams["case_"]].unique()):
        trace = test_preprocessed[test_preprocessed[hparams["case_"]] == idx].reset_index(drop=True)
        trace = trace[trace['start_times']<=split_time]
        running_log = pd.concat([running_log, trace])
    running_log.reset_index(drop=True, inplace=True)
    return running_log

    # trace = test_preprocessed[test_preprocessed[hparams["case_"]] == idx].reset_index(drop=True)
    # df_ = df_[df_[hparams["start_"]] <= split_time]

def generate_rank_indexes(running_log, hparams): #TODO: Da controllare
    
    # Check if there is a rank indexes already created
    if f'rank_indexes.pkl' in os.listdir(f"results/{hparams['exp_name']}"):
        print('Rank indexes already created, read it from the variables folder')
        return pickle.load(open(f"results/{hparams['exp_name']}/rank_indexes.pkl", 'rb'))
    
    print('Start generating rank indexes')
    # Initialize a tuple list with the indexes of the running log and the time predictions
    rank_indexes = list()
    running_log = running_log.iloc[:,:-4].reset_index(drop=True)
    
    # Import the total time predicted before
    model_time = read('results/' + hparams["exp_name"] + '/model_time.pkl')
    for trace_id in tqdm.tqdm(running_log[hparams["case_"]].unique()):
        try:
            trace = running_log[running_log[hparams["case_"]] == trace_id].reset_index(drop=True)
            associated_predicted_time = round(model_time.predict(trace.iloc[-1,1:]), 3)
            rank_indexes.append((trace_id, associated_predicted_time))
        except:
            print('Error in trace ', trace_id)
            
    # Rank it in decreasing order
    rank_indexes = sorted(rank_indexes, key=lambda x: x[1], reverse=True)
    pickle.dump(rank_indexes, open(f"results/{hparams['exp_name']}/rank_indexes.pkl", "wb"))
    print('Rank indexes generated')
    return rank_indexes

def convert_timestamps_to_unix(log, hparams):
    try:
        log[hparams['start_']] = [int(pd.Timestamp(i).timestamp()) for i in log[hparams['start_']].values]
        log[hparams['end_']] = [int(pd.Timestamp(i).timestamp()) for i in log[hparams['end_']].values]
    except:
        log[hparams['timestamp_']] = [int(pd.Timestamp(i).timestamp()) for i in log[hparams['timestamp_']].values]
    return log

def closest_value(target, v):
    differences = np.array(v) - target
    return np.min(differences[differences>0])

def add_activity_duration_column(log, hparams):

    # Check if in the variables folder there is the durations dict
    if f'durations.pkl' in os.listdir(f'variables/{hparams["exp_name"]}'):
        print('Durations already added to the log, read it from the variables folder')
        return pickle.load(open(f'variables/{hparams["exp_name"]}/durations.pkl', 'rb'))

    durations = list()
    for trace_idx in tqdm.tqdm(log[hparams['case_']].unique()):
        
        trace = log[log[hparams['case_']]==trace_idx].reset_index(drop=True)
        vec = np.array(log['start_times'])
        
        for index in trace.index:
        # Search in the timestamp column the closest values to the one in selected line
            try:
                closest_timestamp = closest_value(log['start_times'][index], vec[index:])
                durations.append(closest_timestamp)
            except:
                durations.append(0)

    # Save it to variables folder as pickle file
    pickle.dump(durations, open(f'variables/{hparams["exp_name"]}/durations.pkl', 'wb'))
    print('Durations added to the log')
    
    return durations      

def get_resources_times(log, hparams):
    
    # Check if in the variables folder there is the durations dict
    if f'res_learning_times.pkl' in os.listdir(f'variables/{hparams["exp_name"]}'):
        print('Resources learning times already added to the log, read it from the variables folder')
        return pickle.load(open(f'variables/{hparams["exp_name"]}/res_learning_times.pkl', 'rb'))
    
    rlt = dict()
    
    if '17' in hparams['exp_name']:
        
        print('Generate resources times for BPI17')
        log['durations'] = add_activity_duration_column(log, hparams)        
        for res in tqdm.tqdm(log[hparams['res_']].unique()):
            rlt[res] = dict()
            rlog = log[log[hparams['res_']] == res]
            
            for act in rlog[hparams['act_']].unique():
                rlog_act = rlog[rlog[hparams['act_']] == act]
                
                for timestamp, duration in zip(rlog_act['start_times'], rlog_act['durations']):
                    if act not in rlt[res].keys():
                        rlt[res][act] = []                        
                    rlt[res][act].append((duration, timestamp))
            
        pickle.dump(rlt, open('variables/' + hparams["exp_name"] + '/res_learning_times.pkl', 'wb'))
        return rlt    

    elif 'BAC' in hparams['exp_name']:
        
        print('Generate resources times for BAC')
        for res in tqdm.tqdm(log[hparams['res_']].unique()):
            rlt[res] = dict()
            rlog = log[log[hparams['res_']] == res]
            
            for act in rlog[hparams['act_']].unique():
                rlog_act = rlog[rlog[hparams['act_']] == act]
                
                for idx in rlog_act.index:
                    if act not in rlt[res].keys():
                        rlt[res][act] = [] 
                    rlt[res][act].append((rlog_act[hparams['end_']][idx] - rlog_act[hparams['start_']][idx], rlog_act[hparams['start_']][idx]))
            
        pickle.dump(rlt, open('variables/' + hparams["exp_name"] + '/res_learning_times.pkl', 'wb'))
        return rlt 
        
def filter_on_topk_acts(rec_df, n):
    
    rec_df.reset_index(drop=True, inplace=True)
    ret_df = pd.DataFrame(columns=rec_df.columns)
    
    for act in rec_df['act'].unique():
        alog = rec_df[rec_df['act'] == act].iloc[:n]
        ret_df = pd.concat([ret_df, alog])
     
    return ret_df

def evaluate_Time_Workload_Coefficient(profile, hparams):