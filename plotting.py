#%% 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pickle
import numpy as np
import os
import seaborn as sns
sns.set_style("darkgrid")
import tqdm 
import pandas as pd
import pickle
from datetime import datetime
import json

#%% Helper functions
import utils 

def remove_outliers(vec):
    
    # Drop nan values
    vec = [x for x in vec if str(x) != 'nan']
    
    # Evaluate first-third quartiles
    q1, q3 = np.percentile(vec, [25, 75])
    
    # Evaluate interquartile range (IQR)
    iqr = q3 - q1
    
    # Outlier limits
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter values out of limits
    vec_filtered = [x for x in vec if x >= lower_bound and x <= upper_bound]
    
    return vec_filtered

def rolling_average(data, w=5):
    return np.convolve(data, np.ones(w), 'valid') / w

def convert_to_unix(timestamp_str):
    datetime_object = datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S")
    return int(datetime_object.timestamp())

def plot_activity_durations_with_drift(hparams, w_den=6, drift=None):
    
    """
    w_den is the denominator of the window size for the moving average,
    drift are the starting and ending timestamps of the concept drift"""
    
    act_dur = pickle.load(open(f'results/{hparams["exp_name"]}/res_learning_times.pkl', 'rb'))
    acts_dict = dict()
    for k in act_dur.keys():
        ## if the dictionary not empty
        if act_dur[k]!=dict():    
            for act in act_dur[k].keys():
                if act not in acts_dict.keys():
                    acts_dict[act] = []
                try : acts_dict[act].extend(act_dur[k][act])
                except: print("db")
            
    time_values = []
    for act in acts_dict.keys(): time_values.extend([x[0] for x in acts_dict[act]])
    ranges = [min(time_values),max(time_values)]

    for key in hparams['transition_couples'].keys():
        for act in hparams['transition_couples'][key]:
                couple = key+" - "+act
                cps = acts_dict[couple]
                cps = sorted(cps, key=lambda x: x[1])
                concept_drift = drift[0]
                filter = [x[1]<concept_drift for x in cps]
                cps1 = np.array(cps)[filter]
                w = len(cps1)//w_den
                plt.plot(sorted([x[1] for x in cps1][(w-1):]),  utils.moving_average([x[0] for x in cps1], w=w))
                plt.title(couple+" before concept drift")
                # plt.ylim(ranges)
                plt.savefig(f"results/{hparams['exp_name']}/results_{couple}_before_cd.png")
                plt.close()
                
                
                cps2 = np.array(cps)[np.logical_not(filter)]
                w = len(cps1)//w_den
                # plt.ylim(ranges)
                plt.plot(sorted([x[1] for x in cps2][(w-1):]),  utils.moving_average([x[0] for x in cps2], w=w))
                plt.title(couple+" after concept drift")
                plt.savefig(f"results/{hparams['exp_name']}/results_{couple}_after_cd.png")
                plt.close()
                
    legend_name = []

    for key in hparams['transition_couples'].keys():
        for act in hparams['transition_couples'][key]:
                legend_name.append(key+" - "+act)
                couple = key+" - "+act
                cps1 = acts_dict[couple]
                w = len(cps1)//w_den
                plt.plot(sorted([x[1] for x in cps1][(w-1):]),  utils.moving_average([x[0] for x in cps1], w=w))
    
    plt.axvline(x=drift[0], color='cyan', linestyle='--')
    plt.axvline(x=drift[1], color='cyan', linestyle='--')
    plt.legend(legend_name)
    plt.savefig(f"results/{hparams['exp_name']}/results_all.png")
    plt.close()
    
def plot_activity_durations_without_drift(hparams, w_den=6):
    
    act_dur = pickle.load(open(f'plots/{hparams["exp_name"]}/res_learning_times.pkl', 'rb'))
    acts_dict = dict()
    for k in act_dur.keys():
        ## if the dictionary not empty
        if act_dur[k]!=dict():    
            for act in act_dur[k].keys():
                if act not in acts_dict.keys():
                    acts_dict[act] = []
                try : acts_dict[act].extend(act_dur[k][act])
                except: print("db")
            
    time_values = []
    for act in acts_dict.keys(): time_values.extend([x[0] for x in acts_dict[act]])
    ranges = [min(time_values),max(time_values)]

    for act in hparams['banking_activities']:

        cps = acts_dict[act]
        cps = sorted(cps, key=lambda x: x[1])
        w = len(cps)//w_den
        plt.plot(sorted([x[1] for x in cps][(w-1):]),  utils.moving_average([x[0] for x in cps], w=w))
        plt.title(act+" time distribution")
        # plt.ylim(ranges)
        plt.savefig(f"results/{hparams['exp_name']}/results_{act}.png")
        plt.close()
        
    legend_name = list(acts_dict.keys())
    for act in acts_dict.keys():
        cps = acts_dict[act]
        cps = sorted(cps, key=lambda x: x[1])
        w = len(cps)//w_den
        plt.plot(sorted([x[1] for x in cps][(w-1):]),  utils.moving_average([x[0] for x in cps], w=w))
    plt.ylim([0, 1e9])
    plt.legend(legend_name)
    plt.savefig(f"plots/{hparams['exp_name']}/results_all.png")
    plt.close()    
       
def plot_activities_durations_distribution(log, hparams):

    os.chdir(f"plots/{hparams['exp_name']}")
    
    if 'lifecycles_' in hparams.keys():
        raise NotImplementedError('Not yet implemented for lifecycles')
    
    else:
        # Plot the distribution of the activities durations 
        for act in hparams['banking_activities']:
            plt.figure()
            plt.title(f"duration distribution of {act}")
            plog = log[log[hparams['act_']]==act] 
            durations = plog[hparams['end_']] - plog[hparams['start_']]
            print(durations)
            plt.hist(durations, bins=30)
            plt.savefig(f"dist_{act}.png")
            plt.close()
            
    os.chdir("../..")
     
def plot_traces_lenght_distribution(log, hparams):
     
    os.chdir(f"plots/{hparams['exp_name']}")
    
    durations_ = []
    
    if os.path.exists("plotvars/traces_lenght.pkl"):
        durations_ = pickle.load(open("plotvars/traces_lenght.pkl", "rb"))
        print("Trace analysis found, loaded")
    
    else:
        print("Trace analysis started")
        for trace_id in tqdm.tqdm(log[hparams['case_']].unique()):
            
            p_log = log[log[hparams['case_']]==trace_id]
            
            if 'start_' in hparams.keys():
                start = p_log[hparams['start_']].min()/1000
                end = p_log[hparams['end_']].max()/1000
            
            elif 'timestamp_' in hparams.keys():
                start = p_log[hparams['timestamp_']].min()
                end = p_log[hparams['timestamp_']].max()
            
            durations_.append((start, end))        
        pickle.dump(durations_, open(f"plotvars/traces_lenght.pkl", "wb"))
        print("Trace analysis completed, saved")   
            
    # Convert durations to a Pandas DataFrame
    df = pd.DataFrame(durations_, columns=['Start', 'End'])
    try:
        df['End'] = df['End'].apply(lambda x: convert_to_unix(x))
        df['Start'] = df['Start'].apply(lambda x: convert_to_unix(x))
    except: None
    
    # Calculate the duration for each event
    df['Duration'] = df['End'] - df['Start']
    df.to_csv("plotvars/traces_lenght.csv")
    
    # Convert the timestamps to datetime if they are no datetime type
    if df['Start'].dtype != 'datetime64[ns]':
        df['End'] = pd.to_datetime(df['End'], unit='s')
        df['Start'] = pd.to_datetime(df['Start'], unit='s')
        df['Duration'] = pd.to_datetime(df['Duration'], unit='s')
    
    
        
    # Plot the frequency distribution using Seaborn
    sns.histplot(data=df, x='Start', bins=100, kde=True)
    sns.histplot(data=df, x='End', bins=100, kde=True)
    plt.title('Distribution of Durations All')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    plt.legend(['Starts', 'Ends'])
    exp_name = hparams["exp_name"]
    plt.savefig(f"dist_traces_{exp_name}.png")
    plt.close()
    
    
    # Now plot the active traces

def plot_active_traces_dist(log, hparams):
    
    os.chdir(f"plots/{hparams['exp_name']}")
    
    # Load the durations if present
    if os.path.exists("plotvars/traces_lenght.csv"):
        print("Trace analysis found, loaded")
        df = pd.read_csv("plotvars/traces_lenght.csv", index_col=0) 
    
    else:
        print("Trace analysis not found, loaded")
        plot_traces_lenght_distribution(log, hparams)
        df = pd.read_csv("plotvars/traces_lenght.csv", index_col=0)    
    
    starting_point = df['Start'].min()
    ending_point = df['End'].max()
    
    # divide them in 100 bins
    bins = np.linspace(starting_point, ending_point, 100)
    
    # For each bin, count the number of traces that are active in that bin
    active_traces = []
    for i in range(len(bins)-1):
        active_traces.append(len(df[(df['Start']<bins[i+1]) & (df['End']>bins[i])]))
    
    # Plot the active traces distribution
    plt.plot(bins[:-1], active_traces)
    plt.title(f"Active trace distribution for {hparams['exp_name']} experiment")
    plt.xlabel('Time')
    plt.ylabel('Number of active traces')
    plt.savefig('active_traces_dist.png')
    
    os.chdir("../..")
    
def plot_lenght_distribution(log, hparams):
    
    if not os.path.exists(f"results/{hparams['exp_name']}/traces_lenght.pkl"):
        print("Trace lenght dictionary not found, creating it...")
        utils.create_trace_len_dict(log, hparams)
        print("Trace lenght dictionary created")
        
    plotval = pickle.load(open(f"results/{hparams['exp_name']}/traces_lenght.pkl", "rb"))
    plotval = np.log10(np.array(list(plotval.values())))
    plt.hist(plotval, bins=250)
    plt.title(f"Traces lenght distribution for {hparams['exp_name']} experiment")
    plt.xlabel('LogTime')
    plt.yscale('log')
    plt.ylabel('Log Number of traces')
    plt.savefig(f"plots/{hparams['exp_name']}/traces_lenght_dist.png")
    print("Plot done, saving...")
    
def hours_per_res(dfTrain, timestamp):
    
    for res in simulated_log['org:resource'].unique():            
        log_res = simulated_log[simulated_log['org:resource']==res][['start:timestamp', 'time:timestamp']]
        res_vector[res] = (log_res['time:timestamp']-log_res['start:timestamp']).mean()

    res_vector = pd.DataFrame([res_vector])
    
    
    raise NotADirectoryError("Not yet implemented")
    
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

def plot_mean_and_std(hparams):

    if '17' in hparams['exp_name']:
        # Read the test log
        log_preprocessed =  pd.read_csv(hparams['preprocessed_log_path'])
        split_time = 1480703513 # For BPI17 after
        # log_preprocessed = log_preprocessed[log_preprocessed['start_times']]
        reality_vector = list()
        for res in log_preprocessed['org:resource'].unique():
            reality_vector.append(log_preprocessed[log_preprocessed['org:resource']==res]['time_from_previous_event(start)'].sum())
        reality_vector = sorted(reality_vector)[50:]
        
    else:
        log = pd.read_csv(hparams['log_path'])
        log['duration'] = (log['END_DATE'] - log['START_DATE'])
        test_preprocessed = pd.read_csv(f"logs/{hparams['exp_name']}_dfTest_processed_for_test.csv")
        split_time = test_preprocessed.END_DATE.mean() # - 198567 # For BAC
        log = log[log['END_DATE'] > split_time]
        reality_vector = list()
        for res in log['CE_UO'].unique():
            rlog = log[log['CE_UO']==res]
            reality_vector.append(rlog['duration'].sum())
    reality_vector = np.array(reality_vector)
        # reality_vector = sorted(reality_vector)[20:40]
 
    benchmark = True
    # Read the simulation from the simulations folder
    for i in range(1,11):
        simulated_log = pd.read_csv("simulations/{}/sim{}/sim_{}.csv".format(hparams['exp_name'], '_benchmark'*benchmark, i))
        
        # Convert the timestamp to unix
        simulated_log['start:timestamp'] = simulated_log['start:timestamp'].apply(lambda x: convert_to_unix(x))
        simulated_log['time:timestamp'] = simulated_log['time:timestamp'].apply(lambda x: convert_to_unix(x))
        
        if 'plotting_df_mean' not in vars():
            plotting_df_mean = pd.DataFrame(columns=simulated_log['org:resource'].unique())
            
        res_vector = dict()
        for res in simulated_log['org:resource'].unique():            
            log_res = simulated_log[simulated_log['org:resource']==res][['start:timestamp', 'time:timestamp']]
            res_vector[res] = (log_res['time:timestamp']-log_res['start:timestamp']).mean()

        res_vector = pd.DataFrame([res_vector])
        plotting_df_mean = pd.concat([plotting_df_mean, res_vector], ignore_index=True)

            
        # # Do the same, but for the std
        # if 'plotting_df_std' not in vars():
        #     plotting_df_std = pd.DataFrame(columns=simulated_log['org:resource'].unique())
        
        # res_vector = dict()
        # for res in simulated_log['org:resource'].unique():            
        #     log_res = simulated_log[simulated_log['org:resource']==res][['start:timestamp', 'time:timestamp']]
        #     res_vector[res] = (log_res['time:timestamp']-log_res['start:timestamp']).std()

        # res_vector = pd.DataFrame([res_vector])
        # plotting_df_std = pd.concat([plotting_df_std, res_vector], ignore_index=True)
        
    # Get the mean of every column in the df
    mean_no_bench = plotting_df_mean.mean() 
    
    benchmark = False
    # Read the simulation from the simulations folder
    for i in range(1,11):
        simulated_log = pd.read_csv("simulations/{}/sim{}/sim_{}.csv".format(hparams['exp_name'], '_benchmark'*benchmark, i))
        
        # Convert the timestamp to unix
        simulated_log['start:timestamp'] = simulated_log['start:timestamp'].apply(lambda x: convert_to_unix(x))
        simulated_log['time:timestamp'] = simulated_log['time:timestamp'].apply(lambda x: convert_to_unix(x))
        
        if 'plotting_df_mean' not in vars():
            plotting_df_mean = pd.DataFrame(columns=simulated_log['org:resource'].unique())
            
        res_vector = dict()
        for res in simulated_log['org:resource'].unique():            
            log_res = simulated_log[simulated_log['org:resource']==res][['start:timestamp', 'time:timestamp']]
            res_vector[res] = (log_res['time:timestamp']-log_res['start:timestamp']).mean()

        res_vector = pd.DataFrame([res_vector])
        plotting_df_mean = pd.concat([plotting_df_mean, res_vector], ignore_index=True)
            
            
            
        # Do the same, but for the std
        if 'plotting_df_std' not in vars():
            plotting_df_std = pd.DataFrame(columns=simulated_log['org:resource'].unique())
        
        res_vector = dict()
        for res in simulated_log['org:resource'].unique():            
            log_res = simulated_log[simulated_log['org:resource']==res][['start:timestamp', 'time:timestamp']]
            res_vector[res] = (log_res['time:timestamp']-log_res['start:timestamp']).std()

        res_vector = pd.DataFrame([res_vector])
        plotting_df_std = pd.concat([plotting_df_std, res_vector], ignore_index=True)
    
    mean_bench = plotting_df_mean.mean()
    
    # Plot a boxplot of the mean of the resources using sns
    plt.gca().set_xticklabels(['Fairness-Aware', 'Benchmark', 'Reality'])
    # plt.ylim(-200000/10, 250000)
    plt.ylabel('Mean working time (hours)')
    sns.boxplot(data=[mean_no_bench.values/3600, mean_bench.values/3600, reality_vector/3600], showfliers=False)
    # plt.title('Resources distribution working-times n {} experiment.png'.format(hparams['exp_name']))

    # Save the plot
    plt.savefig("plots/{}/resources distribution working times.png".format(hparams['exp_name']))
    # plt.close()
    
    # Plot a boxplot of the mean of the resources using sns
    
    # plt.gca().set_xticklabels([])
    # sns.boxplot(data=[std_no_bench.values, std_bench.values])
    # plt.title('Resources std distribution {} experiment {}.png'.format(hparams['exp_name'], '_benchmark'*benchmark))
    # plt.legend(['No benchmark', 'Benchmark'])
    
    # # Save the plot
    # plt.savefig("plots/{}/resources_distribution_std_time{}.png".format(hparams['exp_name'], '_benchmark'*benchmark))
    # plt.close()
    print("Plots saved")

                  
# %% Choice the hparameters
hparams = json.load(open("hparams/hparams_bac.json", "r"))
plot_mean_and_std(hparams)
import utils
# %%
# Read the complete log
hparams = json.load(open("hparams/hparams_bpi2017_before.json", "r"))
benchmark = False
log_preprocessed = pd.read_csv(hparams["preprocessed_log_path"])
log_preprocessed = log_preprocessed[[hparams["case_"]] + [col for col in log_preprocessed.columns if col != hparams["case_"]]]
log_preprocessed['duration'] = utils.add_activity_duration_column(log_preprocessed, hparams)
train_indexes = pickle.load(open(f"variables/{hparams['exp_name']}/train_idx.pkl", "rb")) 
test_preprocessed = pd.read_csv(f"logs/{hparams['exp_name']}_dfTest_processed_for_test.csv")
train_preprocessed = pd.read_csv(f"logs/{hparams['exp_name']}_dfTrain_processed_for_train.csv") 
dfTrain = log_preprocessed[log_preprocessed[hparams["case_"]].isin(train_indexes)] 
dfTest = log_preprocessed[~log_preprocessed[hparams["case_"]].isin(train_indexes)] 
# split_time = test_preprocessed.END_DATE.mean() - 198567 # For BAC
# split_time = 1464549028 # For BPI17 before
split_time = 1480703513 # For BPI17 after

nbin = 1000

# %%
# Get the interval 
interval_before = (dfTrain['start_times'].min(), split_time)
interval_after = (split_time, dfTrain['start_times'].max()) 
interval = (dfTrain['start_times'].min(), dfTrain['start_times'].max()) 

# Split the interval in 10000 bins
bins = np.linspace(interval[0], interval[1], nbin)

x_axis_first_part = bins
y_axis_first_part_mean = []
y_axis_first_part_std = []
for bin in tqdm.tqdm(bins):
    rtrain = log_preprocessed[log_preprocessed['start_times']<bin]
    res_dict = dict()
    mean_values = []
    std_values = []
    resources_num = len(log_preprocessed['org:resource'].unique())
    for res in rtrain['org:resource'].unique():
        # mean_values.append(rtrain[rtrain['org:resource']==res]['duration'].mean())
        std_values.append(rtrain[rtrain['org:resource']==res]['duration'].std())
    adds = np.zeros(resources_num - len(std_values))
    if len(adds) != 0:
        resources_num = std_values + list(adds)
    y_axis_first_part_std.append(np.median(std_values))   
w = 100
y_axis_first_part_std = remove_outliers(y_axis_first_part_std)
pickle.dump(y_axis_first_part_std, open(f"variables/{hparams['exp_name']}/std_vector_reality.pkl", "wb"))
sns.lineplot(rolling_average(y_axis_first_part_std, w))
plt.savefig(f"plots/{hparams['exp_name']}/resources_std_distribution_time.png")

# %%
benchmark = True
# Read the simulation from the simulations folder
for i in range(1,11):
    simulated_log = pd.read_csv("simulations/{}/sim{}/sim_{}.csv".format(hparams['exp_name'], '_benchmark'*benchmark, i))
    
    # Convert the timestamp to unix
    simulated_log['start:timestamp'] = simulated_log['start:timestamp'].apply(lambda x: convert_to_unix(x))
    simulated_log['time:timestamp'] = simulated_log['time:timestamp'].apply(lambda x: convert_to_unix(x))
    
    if 'plotting_df_mean' not in vars():
        plotting_df_mean = pd.DataFrame(columns=simulated_log['org:resource'].unique())
        
    res_vector = dict()
    for res in simulated_log['org:resource'].unique():            
        log_res = simulated_log[simulated_log['org:resource']==res][['start:timestamp', 'time:timestamp']]
        res_vector[res] = (log_res['time:timestamp']-log_res['start:timestamp']).std()

    res_vector = pd.DataFrame([res_vector])
    plotting_df_mean = pd.concat([plotting_df_mean, res_vector], ignore_index=True)

w = 25
std_vector = plotting_df_mean.mean()/1000  
pickle.dump(std_vector, open(f"variables/{hparams['exp_name']}/std_vector_benchmark.pkl", "wb")) 
y_axis_second_part_std_bench = remove_outliers(std_vector)
sns.lineplot(rolling_average(y_axis_second_part_std_bench, w))

benchmark = False
# Read the simulation from the simulations folder
for i in range(1,11):
    simulated_log = pd.read_csv("simulations/{}/sim{}/sim_{}.csv".format(hparams['exp_name'], '_benchmark'*benchmark, i))
    
    # Convert the timestamp to unix
    simulated_log['start:timestamp'] = simulated_log['start:timestamp'].apply(lambda x: convert_to_unix(x))
    simulated_log['time:timestamp'] = simulated_log['time:timestamp'].apply(lambda x: convert_to_unix(x))
    
    if 'plotting_df_mean' not in vars():
        plotting_df_mean = pd.DataFrame(columns=simulated_log['org:resource'].unique())
        
    res_vector = dict()
    for res in simulated_log['org:resource'].unique():            
        log_res = simulated_log[simulated_log['org:resource']==res][['start:timestamp', 'time:timestamp']]
        res_vector[res] = (log_res['time:timestamp']-log_res['start:timestamp']).std()

    res_vector = pd.DataFrame([res_vector])
    plotting_df_mean = pd.concat([plotting_df_mean, res_vector], ignore_index=True)

w = 25
std_vector = plotting_df_mean.mean()/1000  
pickle.dump(std_vector, open(f"variables/{hparams['exp_name']}/std_vector_nobenchmark.pkl", "wb")) 
y_axis_second_part_std_bench = remove_outliers(std_vector)
sns.lineplot(rolling_average(y_axis_second_part_std_bench, w))

# %% Same, but for BAC
hparams = json.load(open("hparams/hparams_bac.json", "r"))
log = pd.read_csv(hparams['log_path'])
log['duration'] = (log['END_DATE'] - log['START_DATE'])/1000
test_preprocessed = pd.read_csv(f"logs/{hparams['exp_name']}_dfTest_processed_for_test.csv")
split_time = test_preprocessed.END_DATE.mean() # - 198567 # For BAC
reality_vector = list()
for res in log['CE_UO'].unique():
    rlog = log[log['CE_UO']==res]
    reality_vector.append(rlog['duration'].std())

w = 50
std_vector = remove_outliers(reality_vector) 
pickle.dump(std_vector, open(f"variables/{hparams['exp_name']}/std_vector_reality.pkl", "wb")) 
sns.lineplot(rolling_average(std_vector, w))

# %%
benchmark = True
# Read the simulation from the simulations folder
for i in range(1,11):
    simulated_log = pd.read_csv("simulations/{}/sim{}/sim_{}.csv".format(hparams['exp_name'], '_benchmark'*benchmark, i))
    
    # Convert the timestamp to unix
    simulated_log['start:timestamp'] = simulated_log['start:timestamp'].apply(lambda x: convert_to_unix(x))
    simulated_log['time:timestamp'] = simulated_log['time:timestamp'].apply(lambda x: convert_to_unix(x))
    
    if 'plotting_df_mean' not in vars():
        plotting_df_mean = pd.DataFrame(columns=simulated_log['org:resource'].unique())
        
    res_vector = dict()
    for res in simulated_log['org:resource'].unique():            
        log_res = simulated_log[simulated_log['org:resource']==res][['start:timestamp', 'time:timestamp']]
        res_vector[res] = (log_res['time:timestamp']-log_res['start:timestamp']).std()

    res_vector = pd.DataFrame([res_vector])
    plotting_df_mean = pd.concat([plotting_df_mean, res_vector], ignore_index=True)

w = 25
std_vector = plotting_df_mean.mean()/1000  
pickle.dump(std_vector, open(f"variables/{hparams['exp_name']}/std_vector_benchmark.pkl", "wb")) 
y_axis_second_part_std_bench = remove_outliers(std_vector)
sns.lineplot(rolling_average(y_axis_second_part_std_bench, w))

benchmark = False
# Read the simulation from the simulations folder
for i in range(1,11):
    simulated_log = pd.read_csv("simulations/{}/sim{}/sim_{}.csv".format(hparams['exp_name'], '_benchmark'*benchmark, i))
    
    # Convert the timestamp to unix
    simulated_log['start:timestamp'] = simulated_log['start:timestamp'].apply(lambda x: convert_to_unix(x))
    simulated_log['time:timestamp'] = simulated_log['time:timestamp'].apply(lambda x: convert_to_unix(x))
    
    if 'plotting_df_mean' not in vars():
        plotting_df_mean = pd.DataFrame(columns=simulated_log['org:resource'].unique())
        
    res_vector = dict()
    for res in simulated_log['org:resource'].unique():            
        log_res = simulated_log[simulated_log['org:resource']==res][['start:timestamp', 'time:timestamp']]
        res_vector[res] = (log_res['time:timestamp']-log_res['start:timestamp']).std()

    res_vector = pd.DataFrame([res_vector])
    plotting_df_mean = pd.concat([plotting_df_mean, res_vector], ignore_index=True)

w = 25
std_vector = plotting_df_mean.mean()/1000  
pickle.dump(std_vector, open(f"variables/{hparams['exp_name']}/std_vector_nobenchmark.pkl", "wb")) 
y_axis_second_part_std_bench = remove_outliers(std_vector)
sns.lineplot(rolling_average(y_axis_second_part_std_bench, w))

#0.54, 0.75, per bac w = 40,-19+1sh
# %%
import random
import pm4py
hparams = json.load(open("hparams/hparams_bpi2017_after.json", "r"))
log = pd.read_csv(hparams['preprocessed_log_path'])
reality_vector = pickle.load(open(f"variables/{hparams['exp_name']}/std_vector_reality.pkl", "rb"))
nobench = pickle.load(open(f"variables/{hparams['exp_name']}/std_vector_nobenchmark.pkl", "rb"))
nobench = remove_outliers(nobench)
bench = pickle.load(open(f"variables/{hparams['exp_name']}/std_vector_benchmark.pkl", "rb"))
import random

# %%
w = 5
reality_plot = rolling_average(reality_vector, w)
bench = remove_outliers(bench)
nobench_plot = np.concatenate((reality_plot[:(len(reality_plot) - len(rolling_average(nobench, w)))], rolling_average(nobench, w)))
bench_plot = np.concatenate((reality_plot[:(len(reality_plot) - len(rolling_average(bench, w)))], rolling_average(bench, w)))
nbin = len(reality_plot)

# Get the interval 
# interval = (log['END_DATE'].min(), log['END_DATE'].max())
# interval = (log['END_TIME'].min(), log['END_TIME'].max())
interval = (convert_to_unix(str(log['time:timestamp'].min())), convert_to_unix(str(log['time:timestamp'].max())))
bins = np.linspace(interval[0], interval[1], nbin)
bins = [pd.to_datetime(i, unit='s') for i in bins]
# plt.xlim([bins[200], bins[511]])
sns.lineplot(x = bins, y = nobench_plot)
sns.lineplot(x = bins, y = bench_plot)
sns.lineplot(x = bins, y = reality_plot)
plt.legend(['Workload Distribution', 'Reality', 'Benchmark'])
plt.ylabel('Std of working times')
plt.xticks(rotation=45) 
plt.title(f'{hparams["exp_name"]}')
plt.savefig(f"plots/{hparams['exp_name']}/resources_std_distribution_time.png")
# %%
    
