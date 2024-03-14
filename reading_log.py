import pandas as pd
import pm4py 
import numpy as np
import os

import IO

def convert_to_csv(filename=str):
    if '.csv' in filename:
        return None
    if '.xes.gz' in filename:
        pm4py.convert_to_dataframe(pm4py.read_xes(filename)).to_csv(path_or_buf=(filename[:-7] + '.csv'), index=None)
        print('Conversion ok')
        return None
    if '.xes' in filename:
        pm4py.convert_to_dataframe(pm4py.read_xes(filename)).to_csv(path_or_buf=(filename[:-4] + '.csv'), index=None)
        print('Conversion ok')
    else:
        raise TypeError('Check the path or the log type, admitted formats : csv, xes, xes.gz')


def modify_filename(filename):
    if '.csv' in filename: return filename
    if '.xes.gz' in filename: return filename[:-7] + '.csv'
    if '.xes' in filename:
        return filename[:-4] + '.csv'
    else:
        None


def read_data(filename, start_time_col, date_format="%Y-%m-%d %H:%M:%S"):
    if '.csv' in filename:
        try:
            df = pd.read_csv(filename, header=0, low_memory=False)
            # if df.columns[0]=='Unnamed: 0':
            #     df = df.iloc[:,1:]
        except UnicodeDecodeError:
            df = pd.read_csv(filename, header=0, encoding="cp1252", low_memory=False)
    elif '.parquet' in filename:
        df = pd.read_parquet(filename, engine='pyarrow')
    # if a datetime cast it to seconds
    if not np.issubdtype(df[start_time_col], np.number):
        df[start_time_col] = pd.to_datetime(df[start_time_col], format=date_format)
        df[start_time_col] = df[start_time_col].astype(np.int64) / int(1e9)
    return df


def read_log(params):
    
    save = False
    if not '.csv' in params['log_path']:
        #Check if there's a .csv copy
        if not os.path.exists(params['log_path'][:-4] + '.csv'):
            save = True
        else: 
            print('Reading from csv copy')
            return pd.read_csv(params['log_path'][:-4] + '.csv')
        
    convert_to_csv(filename=params['log_path'])
    filename = modify_filename(params['log_path'])
    df = read_data(params['log_path'], params['end_'], params['date_'])
    if save:
        IO.write_csv(df, filename)
    return df

def read_preprocessed_log(hparams):
    
    return pd.read_csv(hparams['preprocessed_log_path'])