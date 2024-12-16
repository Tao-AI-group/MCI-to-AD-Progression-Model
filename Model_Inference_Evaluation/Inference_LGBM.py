
# Basic imports
import warnings
warnings.filterwarnings('ignore')

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../libs/'))
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import math
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import def_function as func

# ML metrics
import sklearn.metrics as m
from sklearn.utils import resample

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve,
    classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer


from lightgbm import LGBMClassifier


def get_common_path(relative_path):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(cur_path, relative_path)
    return file_path


def time_consumption_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def prepare_train_and_test_features(train_sl: List, valid_sl: List, test_sl: List, input_size_1: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare features and labels for model training and testing."""
    # Process training and validation data
    pts_tr, labels_tr, features_tr = [], [], []
    for pt in train_sl + valid_sl:
        pts_tr.append(pt[0])
        labels_tr.append(pt[1])
        features_tr.append([code for v in pt[-1] for code in v[-1]])
    
    # Process test data
    pts_t, labels_t, features_t = [], [], []
    for pt in test_sl:                  
        pts_t.append(pt[0])
        labels_t.append(pt[1])
        features_t.append([code for v in pt[-1] for code in v[-1]])

    mlb = MultiLabelBinarizer(classes=range(input_size_1[0])[1:])
    return mlb.fit_transform(features_tr), np.array(labels_tr), mlb.transform(features_t), np.array(labels_t)

def load_input_pkl(common_path):
    """Load train, test, validation and type mapping data from pickle files."""
    pkl_result_list = []
    input_name_list = ['.combined.train', '.combined.test', '.combined.valid', '.types']   
    
    for name in input_name_list:
        input_path = common_path + "ad" + name
        pkl_result = pickle.load(open(input_path, 'rb'), encoding='bytes')
        pkl_result_list.append(pkl_result)
        
    return pkl_result_list[0], pkl_result_list[1], pkl_result_list[2], pkl_result_list[3]


if __name__ == "__main__":
    start_tm = datetime.now()
    output_folder_name = 'Results_15_final_new'
    days_before_index_date = 1825
    part_common_path = get_common_path('../../../Results_new_no_censored/') + output_folder_name + '/'
    results_output_path = func.get_common_path('../../Test_results_no_censored/') + '/' + str(days_before_index_date) + '/'
    
    selected_list = ['Results_1d_365d_window']
    #selected_list = ['Results_1d_730d_window']
    #selected_list = ['Results_1d_1095d_window']
    #selected_list = ['Results_1d_1825d_window']
  
    tuning_results_list = []
    para_value_list = []
    para_name_list = []

    for idx_folder, specific_folder_name in enumerate(selected_list):
        common_path = part_common_path + str(days_before_index_date) +  "/" + specific_folder_name + '/'        
        print(f'{idx_folder+1}. Running ML for file from {specific_folder_name}:', '\n')  
        
        # Load data
        train_sl, test_sl, valid_sl, types_d = load_input_pkl(common_path)
        types_d_rev = dict(zip(types_d.values(), types_d.keys()))

        # Prepare features
        input_size_1 = [len(types_d_rev) + 1]
        nfeatures_tr, labels_tr, nfeatures_t, labels_t = prepare_train_and_test_features(train_sl, valid_sl, test_sl, input_size_1)
        
          
        to_infere_LGBM_results_label = 1
        if  to_infere_LGBM_results_label:
            
            # Fit the best model on the train+validation data and save the model
            
            # Define optimal prameters
            params ={
                "n_estimators" :  1000,
                "learning_rate" : 0.014112966741633416,
                "max_depth" : 9,
                "reg_alpha" : 25,
                "reg_lambda" : 50,
                "num_leaves" : 760,
                "feature_fraction" : 0.6,
                "bagging_fraction" : 0.9,
                "bagging_freq" : 1,
                "min_child_samples" : 100}
            
            # Initialize and train model
            clf = LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                class_weight="balanced",
                **params
            )
            clf.fit(nfeatures_tr, labels_tr)
            
            # Save model
            ckpt_dir = os.path.join(results_output_path, 'optuna_tuning', 'objective_LGBM_MCI', '1d_365d')
            if not os.path.exists(ckpt_dir):
                print("Creating checkpoint directory")
                os.makedirs(ckpt_dir)
            
            model_path = os.path.join(ckpt_dir, 'test_model.p')
            func.save_pkl(clf, model_path)
            
            # Generate predictions
            y_score = clf.predict_proba(nfeatures_t)[:,1]
            y_pred = clf.predict(nfeatures_t)
            
            # Print classification report
            print('\nClassification Report:')
            print(classification_report(labels_t, y_pred))
            
            #Calculate all measures
            auc= roc_auc_score(labels_t, y_score)
            auprc = average_precision_score(labels_t, y_score)
            precision, recall, fscore, support = precision_recall_fscore_support(labels_t, y_pred, average = 'binary')
            precision_weighted, recall_weighted, fscore_weighted, _ = \
                precision_recall_fscore_support(labels_t, y_pred, average='weighted')

            
        # Bootstrap performance evaluation 
        to_calculate_bootstrapped_performance = 0
        if to_calculate_bootstrapped_performance:
            
            # Load saved model
            model_path = os.path.join(results_output_path, 'optuna_tuning', 'objective_LGBM_MCI', '1d_365d', 'test_model.p')
            clf = pickle.load(open(model_path, 'rb'))
            
            
            # For AUPRC CUrve
            y_score = clf.predict_proba(nfeatures_t)[:,1]
            precision, recall, _ = precision_recall_curve(labels_t, y_score)
            
            
            # For AUPRC Curve
            for data, name in [(precision, 'precision'), (recall, 'recall')]:
                with open(os.path.join(results_output_path, f'{name}_LGBM_1d_365d.pkl'), 'wb') as f:
                    pickle.dump(data.tolist(), f)
                
            
            # Perform bootstrap analysis
            n_trials = 500
            alpha = 0.05
            all_CI = defaultdict(list)
            
            print("Starting bootstrap trials...")
            for trial_num in range(n_trials):
                # Resample test data
                X_b, y_b = resample(nfeatures_t, labels_t)
                
                # Generate predictions
                y_score = clf.predict_proba(X_b)[:,1]
                y_pred = clf.predict(X_b)
                
                # Calculate metrics
                precision, recall, fscore, _ = precision_recall_fscore_support(y_b, y_pred, average='weighted')
                tn, fp, fn, tp = confusion_matrix(y_b, y_pred).ravel()
                specificity = tn / (tn + fp)
                
                # Store all metrics
                metrics = {
                    "auc": roc_auc_score(y_b, y_score),
                    "auprc": average_precision_score(y_b, y_score),
                    "precision": precision_score(y_b, y_pred),
                    "recall": recall_score(y_b, y_pred),
                    "f1": f1_score(y_b, y_pred),
                    "precision_weighted": precision,
                    "recall_weighted": recall,
                    "f1_weighted": fscore,
                    "specificity": specificity,
                    "accuracy": accuracy_score(y_b, y_pred),
                    "balanced accuracy": balanced_accuracy_score(y_b, y_pred)
                }
                
                for key, value in metrics.items():
                    all_CI[key].append(value)
                
                print(f"Completed bootstrap trial {trial_num + 1}/{n_trials}")
            
            # Calculate confidence intervals
            df = pd.DataFrame(columns=["lower", "upper", "95%_CI", "stat", "model"])
            
            for key in all_CI.keys():
                lower = round(max(0, np.percentile(all_CI[key], (alpha/2) * 100)), 4)
                upper = round(min(1, np.percentile(all_CI[key], (1-alpha/2) * 100)), 4)
                
                df.loc[len(df)] = [
                    lower,
                    upper,
                    f"({lower},{upper})",
                    key,
                    os.path.splitext(os.path.basename('1d_365d'))[0]
                ]
            
            df.to_csv("confidence_interval_LGBM_1d_365d.csv", index=False)
            
    print(f'Finished running all models, total time: {datetime.now() - start_tm}')
    print('Optimization complete!')
