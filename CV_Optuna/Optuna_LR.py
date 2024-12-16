import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import numpy as np
import multiprocessing
from datetime import datetime
from typing import List, Tuple, Dict, Any

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support,
    accuracy_score, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

import optuna

def get_optimal_cpu_count(max_percent: int = 80) -> int:
    """Calculate optimal number of CPU cores to use."""
    total_cores = multiprocessing.cpu_count()
    target_cores = int(total_cores * (max_percent / 100))
    return max(1, min(target_cores, total_cores - 1))

def get_common_path(relative_path: str) -> str:
    """Generate absolute path from relative path."""
    cur_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cur_path, relative_path)

def load_input_pkl(common_path: str) -> Tuple[List, List, List, Dict]:
    """Load data from pickle files."""
    pkl_result_list = []
    input_name_list = ['.combined.train4', '.combined.test4', '.combined.valid4', '.types']   
    for name in input_name_list:
        input_path = common_path + "lt" + name
        pkl_result = pickle.load(open(input_path, 'rb'), encoding='bytes')
        pkl_result_list.append(pkl_result)
    return tuple(pkl_result_list)

def prepare_features(train_sl: List, valid_sl: List, test_sl: List, input_size_1: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare features for model training."""
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
    return mlb.fit_transform(features_tr), np.array(labels_tr), mlb.fit_transform(features_t)

def objective_LR(trial):
    """Optuna objective function for Logistic Regression optimization."""
    # Configure CPU usage
    n_cores = get_optimal_cpu_count(max_percent=50)
    threading_cores = max(1, n_cores // 2)
    
    # Set threading environment variables
    for var in ['MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']:
        os.environ[var] = str(threading_cores)
    
    # Define hyperparameters
    logreg_c = trial.suggest_float("logreg_c", 1e-10, 1e10, log=True)
    
    # Initialize metrics
    metrics = {m: np.empty(5) for m in ['auc', 'ap', 'precision', 'recall', 'f1', 'acc', 'balanced_acc']}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
    labels_tr_array = np.array(labels_tr)
    
    for idx, (train_idx, val_idx) in enumerate(cv.split(nfeatures_tr, labels_tr_array)):
        print(f"Training fold {idx+1}/5 using {n_cores} CPU cores...")
        
        # Split data
        X_train, X_val = nfeatures_tr[train_idx], nfeatures_tr[val_idx]
        y_train, y_val = labels_tr_array[train_idx], labels_tr_array[val_idx]

        # Train and predict
        clf = LogisticRegression(C=logreg_c, random_state=47)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_val)[:,1]
        y_pred = clf.predict(X_val)
        
        # Calculate metrics
        metrics['auc'][idx] = roc_auc_score(y_val, y_score)
        metrics['ap'][idx] = average_precision_score(y_val, y_score)
        metrics['precision'][idx], metrics['recall'][idx], metrics['f1'][idx], _ = \
            precision_recall_fscore_support(y_val, y_pred, average="binary")
        metrics['acc'][idx] = accuracy_score(y_val, y_pred)
        metrics['balanced_acc'][idx] = balanced_accuracy_score(y_val, y_pred)
    
    # Log results
    print(f'AUC: {np.mean(metrics["auc"]):.4f}, AP: {np.mean(metrics["ap"]):.4f}')
    print(f'Precision: {np.mean(metrics["precision"]):.4f}, Recall: {np.mean(metrics["recall"]):.4f}, F1: {np.mean(metrics["f1"]):.4f}')
    print(f'Accuracy: {np.mean(metrics["acc"]):.4f}, Balanced Accuracy: {np.mean(metrics["balanced_acc"]):.4f}\n')
    
    results_list.append([np.mean(metrics['auc']), np.mean(metrics['ap'])])
    return np.mean(metrics['auc'])

if __name__ == "__main__":
    start_tm = datetime.now()

    # Print CPU configuration
    total_cores = multiprocessing.cpu_count()
    used_cores = get_optimal_cpu_count(50)
    print(f"Total CPU cores available: {total_cores}")
    print(f"Using {used_cores} cores (targeting 50% CPU usage)")
    print(f"Reserved cores for system: {total_cores - used_cores}")

    # Initialize paths and configuration
    output_folder_name = 'Results_15_final_new'
    days_before_index_date = 1825
    part_common_path = get_common_path('../../../Results_new_no_censored/') + output_folder_name + '/'
    
    selected_list = ['Results_1d_1825d_window']
    specific_folder_name = selected_list[0]
    #selected_list = ['Results_1d_730d_window']
    #selected_list = ['Results_1d_1095d_window']
    #selected_list = ['Results_1d_1825d_window']

    tuning_results_list = []
    para_value_list = []
    para_name_list = []

    for idx_folder, specific_folder_name in enumerate(selected_list):
        common_path = part_common_path + str(days_before_index_date) + "/" + specific_folder_name + '/'        
        print(f'{idx_folder+1}. Running ML for file from {specific_folder_name}:', '\n')  
        
        # Load and preprocess data
        train_sl, test_sl, valid_sl, types_d = load_input_pkl(common_path)
        types_d_rev = dict(zip(types_d.values(), types_d.keys()))
        input_size_1 = [len(types_d_rev) + 1]
        

        # Prepare features
        nfeatures_tr, labels_tr, nfeatures_t = prepare_features(train_sl, valid_sl, test_sl, input_size_1)
        
        # Run optimization
        print(f'{idx_folder+1}. Start LR hyperparameter tuning for {specific_folder_name}, time is {datetime.now()}', '\n')
        results_list = []
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_LR, n_trials=100)

        # Log results
        print("\nOptimization Results:")
        print(f"Number of finished trials: {len(study.trials)}")
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Trial Number: {trial.number}")
        print(f"  Value: {trial.value:.4f}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            para_value_list.append(value)
            if len(para_name_list) < len(trial.params):
                para_name_list.append(key)

        best_result = results_list[trial.number]
        tuning_result_row = [specific_folder_name, 'LR'] + best_result + [trial.number, trial.params] + para_value_list
        print(f'tuning_result_row: {tuning_result_row}')
        tuning_results_list.append(tuning_result_row)

    print(f'Finished running all ML optimizations, total time: {datetime.now() - start_tm}')
    print('Optimization complete!')