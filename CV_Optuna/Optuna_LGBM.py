# Basic imports
import warnings
warnings.filterwarnings('ignore')

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../libs/'))
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Any

# ML metrics
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_fscore_support,
    accuracy_score, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer


from lightgbm import LGBMClassifier
import optuna


def get_common_path(relative_path):
    """Convert relative path to absolute path."""
    cur_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cur_path, relative_path)


def load_input_pkl(common_path):
    """Load train, test, validation and type mapping data from pickle files."""
    pkl_result_list = []
    input_name_list = ['.combined.train', '.combined.test', '.combined.valid', '.types']   
    
    for name in input_name_list:
        input_path = common_path + "ad" + name
        pkl_result = pickle.load(open(input_path, 'rb'), encoding='bytes')
        pkl_result_list.append(pkl_result)
        
    return pkl_result_list[0], pkl_result_list[1], pkl_result_list[2], pkl_result_list[3]

def prepare_features(train_sl: List, valid_sl: List, test_sl: List, input_size_1: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare features for model training."""
    # Process training and validation data
    pts_tr, labels_tr, features_tr = [], [], []
    for pt in train_sl + valid_sl:
        pts_tr.append(pt[0])
        labels_tr.append(pt[1])
        x = []
        for v in pt[-1]:
            x.extend(v[-1])
        features_tr.append(x)
    
    # Process test data
    pts_t, labels_t, features_t = [], [], []
    for pt in test_sl:                  
        pts_t.append(pt[0])
        labels_t.append(pt[1])
        x = []
        for v in pt[-1]:
            x.extend(v[-1])
        features_t.append(x)    

    mlb = MultiLabelBinarizer(classes=range(input_size_1[0])[1:])
    return mlb.fit_transform(features_tr), np.array(labels_tr), mlb.fit_transform(features_t)

def objective_LGBM(trial):
    """Optuna objective function for LightGBM optimization."""
    param_grid = {
        "n_estimators" :  trial.suggest_categorical("n_estimators", [1000]),
        "learning_rate" : trial.suggest_float("learning_rate", 0.001, 0.3),
        "max_depth" : trial.suggest_int("max_depth", 3,10),
        "reg_alpha" : trial.suggest_int("reg_alpha", 0, 100, step=5),
        "reg_lambda" : trial.suggest_int("reg_lambda", 0, 100, step=5),
        "num_leaves" : trial.suggest_int("num_leaves", 20, 1000, step=20),
        "feature_fraction" : trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
        "bagging_fraction" : trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
        "bagging_freq" : trial.suggest_categorical("bagging_freq", [1]),
        "min_child_samples" : trial.suggest_int("min_child_samples", 100, 1000, step=50)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 47)
    metrics = {m: np.empty(5) for m in ['auc', 'ap', 'precision', 'recall', 'f1', 'acc', 'balanced_acc']}
    
    for idx, (train_idx, val_idx) in enumerate(cv.split(nfeatures_tr, labels_tr)):
        X_train, X_val = nfeatures_tr[train_idx], nfeatures_tr[val_idx]
        y_train, y_val = labels_tr[train_idx], labels_tr[val_idx]

        clf = LGBMClassifier(
            objective='binary',
            random_state=47,
            class_weight="balanced",
            **param_grid
        )
        
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_val)[:,1]
        y_pred = clf.predict(X_val)
        
        metrics['auc'][idx] = roc_auc_score(y_val, y_score)
        metrics['ap'][idx] = average_precision_score(y_val, y_score)
        metrics['precision'][idx], metrics['recall'][idx], metrics['f1'][idx], _ = \
            precision_recall_fscore_support(y_val, y_pred, average="binary")
        metrics['acc'][idx] = accuracy_score(y_val, y_pred)
        metrics['balanced_acc'][idx] = balanced_accuracy_score(y_val, y_pred)
    
    print(f'auc_test is {np.mean(metrics["auc"])}, ap_test is {np.mean(metrics["ap"])}')
    results_list.append([np.mean(metrics['auc']), np.mean(metrics['ap'])])
    print(f'precision: {np.mean(metrics["precision"])}, recall: {np.mean(metrics["recall"])},\
          fscore: {np.mean(metrics["f1"])}', '\n')
    print(f'accuracy: {np.mean(metrics["acc"])}, balanced accuracy: {np.mean(metrics["balanced_acc"])}', "\n")

    return np.mean(metrics['auc'])


if __name__ == "__main__":
    start_tm = datetime.now()
    output_folder_name = 'Results_15_final_new'
    days_before_index_date = 1825
    part_common_path = get_common_path('../../../Results_new_no_censored/') + output_folder_name + '/'
    
    
    print('Run trained LGBM and get the best model')
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
        nfeatures_tr, labels_tr, nfeatures_t = prepare_features(train_sl, valid_sl, test_sl, input_size_1)
        
        
        # Optimize model
        print(f'Starting LGBM hyperparameter tuning for {specific_folder_name} at {datetime.now()}')
        results_list = []

        study = optuna.create_study(direction="maximize")
        study.optimize(objective_LGBM, n_trials=100)
     
        # Log results
        print("\nOptimization Results:")
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Trial Number: {best_trial.number}")
        print(f"  Value: {best_trial.value:.4f}")
        print("  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
            para_value_list.append(value)
            if len(para_name_list) < len(best_trial.params):
                para_name_list.append(key)
        
        best_result = results_list[best_trial.number]
        tuning_result_row = [specific_folder_name, 'LGBM'] + best_result + [best_trial.number, best_trial.params] + para_value_list
        tuning_results_list.append(tuning_result_row)


    print(f'Finished running all ML optimizations, total time: {datetime.now() - start_tm}')
    print('Optimization complete!')