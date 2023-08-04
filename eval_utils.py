from pycox.models import CoxPH, DeepHitSingle, CoxTime, PCHazard
from sksurv.linear_model import CoxnetSurvivalAnalysis
import os
import pickle
from pycox.evaluation import EvalSurv
import torch
import torchtuples as tt
import os
import pickle
from lifelines.utils import concordance_index

import numpy as np
import pandas as pd

np.random.seed(42)
_ = torch.manual_seed(42)


DL_MODELS = ["deep_surv", "pc_hazard", "cox_time", "deep_hit"]

ML_MODELS = ["reg_coxph", "rsf", "gbst"]

MODELS = DL_MODELS + ML_MODELS


MODEL_CLASS = {'deep_surv': CoxPH, 
               'pc_hazard': PCHazard,
               'deep_hit': DeepHitSingle,
               'cox_time': CoxTime,
              'reg_coxph' : CoxnetSurvivalAnalysis
            }

def get_results(file_name, model_name, X, y, X_test, y_test):
    
    if model_name in DL_MODELS:
        path_sc = os.path.join(f"statistics/{model_name}/models", f"sc_{file_name}")
        path_net = os.path.join(f"statistics/{model_name}/models", f"net_{file_name}").replace(".pkl", "")
        path_weights = os.path.join(f"statistics/{model_name}/models", f"weights_{file_name}").replace(".pkl", "")

        # load the standar scaler
        sc = pickle.load(open(path_sc, 'rb'))
        
        # initilize a generic nn that will be override
        net = tt.practical.MLPVanilla(2, [41], 1, False, 0.1)

        # load the model
        model = MODEL_CLASS[model_name](net)
        model.load_net(path_net)
        model.load_model_weights(path_weights)                        

        # model evaluation
        original_y = y.copy()
        original_y_test = y_test.copy()
        
        y = y["duration"].values, y["event"].values        
        X = sc.transform(X).astype('float32')
        X_test = sc.transform(X_test).astype('float32')
        y_test = y_test["duration"].values, y_test["event"].values
        
        ## pycox measures
        if model_name == 'pc_hazard':
            model.sub = 10
            estimate_surv = model.predict_surv_df(X_test)
        
        if model_name == 'deep_hit':
            estimate_surv = model.interpolate(10).predict_surv_df(X_test)
            
        if model_name == 'deep_surv':
            estimate_surv = model.predict_surv_df(X_test)
        
        if model_name == 'cox_time':
            estimate_surv = model.predict_surv_df(X_test)
            
        ev = EvalSurv(estimate_surv, y_test[0], y_test[1], censor_surv='km')

        tmp_times = np.sort(y[0])
        times = np.array([tmp_times[i] for i in range(0, len(tmp_times), 10)])

        concordance_td = ev.concordance_td('antolini')
        ibs = ev.integrated_brier_score(times)

        ## lifelines measures
        estimate = np.mean(1-estimate_surv, axis=0)
        c_index = 1 - concordance_index(event_times=original_y_test["duration"], 
                          predicted_scores= estimate, 
                          event_observed=original_y_test["event"])
        
        return c_index, concordance_td, ibs
    
    if model_name in ML_MODELS:
        path_sc = os.path.join(f"statistics/{model_name}/models", f"sc_{file_name}")
        path_model = os.path.join(f"statistics/{model_name}/models", f"model_{file_name}").replace(".pkl", "")

        # load the standar scaler
        sc = pickle.load(open(path_sc, 'rb'))

        # load the model
        model = pickle.load(open(path_model, 'rb'))
    
        # model evaluation        
        X = sc.transform(X).astype('float32')
        X_test = sc.transform(X_test).astype('float32')
        
        estimate = model.predict(X_test)
        
        ## pycox measures
        original_y = y["duration"].values, y["event"].values                
        original_y_test = y_test["duration"].values, y_test["event"].values        
        
        estimate_surv = model.predict_survival_function(X_test, return_array=True)
        estimate_surv = pd.DataFrame(estimate_surv.T)
        estimate_surv.index = model.event_times_

        ev = EvalSurv(estimate_surv, original_y_test[0], original_y_test[1], censor_surv='km')

        tmp_times = np.sort(original_y[0])
        times = np.array([tmp_times[i] for i in range(0, len(tmp_times), 10)])

        concordance_td = ev.concordance_td('antolini')

        ibs = ev.integrated_brier_score(times)

        ## lifelines measures
        c_index = 1 - concordance_index(event_times = original_y_test[0], 
                          predicted_scores = estimate, 
                          event_observed = original_y_test[1])
                        
        return c_index, concordance_td, ibs