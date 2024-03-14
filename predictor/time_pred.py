# %%
import catboost
import numpy as np
import json
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from catboost.utils import select_threshold
from IO import read, write

# d = json.load(open("/home/padella/Desktop/gui_catboost/experiments/bac_time4split--10-01-2024_11-00-47_409719+0000/model/data_info.json"))

def fit_model(train_df, y, hparams):

    case_id_name, activity_name, experiment_name = hparams["case_"], hparams["act_"], hparams["exp_name"]

    categorical_features = train_df.select_dtypes(exclude=np.number).columns
    train_df[categorical_features] = train_df[categorical_features].astype(str)
    column_types = train_df.dtypes.astype(str).to_dict()
    
    params = {
        'depth': 12,
        'learning_rate': 0.1,
        'iterations': 25000,
        'early_stopping_rounds': 250,
        'thread_count': 4,
        'logging_level': 'Verbose',
        'task_type': "CPU"  # "GPU" if int(os.environ["USE_GPU"]) else "CPU"
    }

    print('Starting training...')
    params["loss_function"] = "MAE"
    train_data = Pool(train_df, y, cat_features=categorical_features.values)
    model = CatBoostRegressor(**params)
    model.fit(train_data, verbose=True, plot=True, eval_set=(train_data))
    return model
    
def predict_model(train_df, y, model, hparams):
    
    case_id_name, activity_name, experiment_name = hparams["case_"], hparams["act_"], hparams["exp_name"]

    categorical_features = train_df.select_dtypes(exclude=np.number).columns
    train_df[categorical_features] = train_df[categorical_features].astype(str)
    column_types = train_df.dtypes.astype(str).to_dict()
    
    #predict the train score 
    train_data = Pool(train_df, cat_features=categorical_features.values)
    y_pred = model.predict(train_data)
    
    #evaluate mae
    mae = np.mean(np.abs(y_pred - train_df[activity_name].values))
    return mae
    
def testing_model(test_df, model, hparams):
    case_id_name, activity_name, experiment_name = hparams["case_"], hparams["act_"], hparams["exp_name"]
    categorical_features = test_df.select_dtypes(exclude=np.number).columns
    test_df[categorical_features] = test_df[categorical_features].astype(str)
    column_types = test_df.dtypes.astype(str).to_dict()
    test_data = Pool(test_df, cat_features=categorical_features.values)
    y_pred = model.predict(test_data)
    return y_pred

def preprocess_and_train(train_df, hparams, model_name):
      
    if model_name == 'model1':
        X_train = train_df.iloc[:, 1:-4]
        y_train = train_df.iloc[:, -4]
        model = fit_model(X_train, y_train, hparams)
        
        
    if model_name == 'model5':
        X_train = train_df.iloc[:, 1:-4]
        y_train = train_df.iloc[:, -3]
        model = fit_model(X_train, y_train, hparams)
        
    if model_name == 'model20':
        X_train = train_df.iloc[:, 1:-4]
        y_train = train_df.iloc[:, -2]            
        model = fit_model(X_train, y_train, hparams)
        
    if model_name == 'time':
        X_train = train_df.iloc[:, 1:-4]
        y_train = train_df.iloc[:, -1]
        model = fit_model(X_train, y_train, hparams)
    
    return model

def import_predictor(hparams, model_num=1):
    model = catboost.CatBoostRegressor()
    model = read('results/' + hparams["exp_name"] + f'/model{model_num}.pkl')

    return model

def apply_pred(last_event, act_res_tuple, model, hparams):
    act = act_res_tuple[0][0]
    res = act_res_tuple[0][1]
    last_event[hparams["act_"]] = act
    last_event[hparams["res_"]] = res
    return np.abs(model.predict([last_event.values[1:-1]])[0])
