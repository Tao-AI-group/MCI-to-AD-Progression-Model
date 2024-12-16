### Basic imports
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
np.random.seed(41)
import random
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../libs/'))
import math
import pickle
import os
from datetime import datetime
import time

# ML metrics
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support
)

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Custom imports
import models as model 
from EHRDataloader import EHRdataloader
from EHRDataloader import EHRdataFromLoadedPickles as EHRDataset
import utils as ut 
import def_function as func

np.random.seed(41)

# Hyperparameter optimization
import optuna

WEIGHT_CLASS_0 = None  # Weight for majority class (zeros)
WEIGHT_CLASS_1 = None  # Weight for minority class (ones)

# Configure CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def get_common_path(relative_path):
    """Convert relative path to absolute path."""
    cur_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cur_path, relative_path)

def time_consumption_since(since):
    """Calculate time elapsed in minutes and seconds."""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def load_input_pkl(common_path):
    """Load train, test, validation and type mapping data from pickle files."""
    pkl_result_list = []
    input_name_list = ['.combined.train', '.combined.test', '.combined.valid', '.types']   
    
    for name in input_name_list:
        input_path = common_path + "ad" + name
        pkl_result = pickle.load(open(input_path, 'rb'), encoding='bytes')
        pkl_result_list.append(pkl_result)
        
    return pkl_result_list[0], pkl_result_list[1], pkl_result_list[2], pkl_result_list[3]

def trainbatches(mbs_list, model, optimizer, shuffle=True, loss_fn=nn.BCELoss()): 
    """Train model on batches with specified loss function and optimizer."""
    global WEIGHT_CLASS_0, WEIGHT_CLASS_1

    current_loss = 0
    all_losses = []
    plot_every = 5
    n_iter = 0 
    
    if shuffle: 
        random.shuffle(mbs_list)
        
    for i, batch in enumerate(mbs_list):
        sample, label_tensor, seq_l, mtd = batch
        
        # Set class weights using global variables
        weight = torch.zeros_like(label_tensor)
        weight[label_tensor==0] = WEIGHT_CLASS_0
        weight[label_tensor==1] = WEIGHT_CLASS_1
        
        loss_fn = nn.BCELoss(weight.squeeze()) 
        
        # Train on single batch
        output, loss = ut.trainsample(sample, label_tensor, seq_l, mtd, model, optimizer, criterion=loss_fn)
        current_loss += loss
        n_iter += 1
    
        if n_iter % plot_every == 0:
            all_losses.append(current_loss/plot_every)
            current_loss = 0    
            
    return current_loss, all_losses 

def run_dl_model(ehr_model, train_mbs, valid_mbs, wmodel, epochs=100, 
                l2=0.0001, lr=0.0005, eps=1e-4, opt='Adagrad', patience=5):
    """Train deep learning model with early stopping."""
    # Initialize optimizer based on specified type
    if opt == 'Adadelta':
        optimizer = optim.Adadelta(ehr_model.parameters(), lr=lr, weight_decay=l2, eps=eps)
    elif opt == 'Adagrad':
        optimizer = optim.Adagrad(ehr_model.parameters(), lr=lr, weight_decay=l2)
    elif opt == 'Adam':
        optimizer = optim.Adam(ehr_model.parameters(), lr=lr, weight_decay=l2, eps=eps)
    elif opt == 'Adamax':
        optimizer = optim.Adamax(ehr_model.parameters(), lr=lr, weight_decay=l2, eps=eps)
    elif opt == 'RMSprop':
        optimizer = optim.RMSprop(ehr_model.parameters(), lr=lr, weight_decay=l2, eps=eps)
    elif opt == 'ASGD':
        optimizer = optim.ASGD(ehr_model.parameters(), lr=lr, weight_decay=l2)
    elif opt == 'SGD':
        optimizer = optim.SGD(ehr_model.parameters(), lr=lr, weight_decay=l2, momentum=0.9)

    # Training epochs with early stopping
    bestValidAuc = 0.0
    bestValidEpoch = 0

    for ep in range(epochs):
        # Train
        start = time.time()
        current_loss, train_loss = trainbatches(train_mbs, model=ehr_model, optimizer=optimizer)
        avg_loss = np.mean(train_loss)
        train_time = time_consumption_since(start)
        
        # Evaluate
        eval_start = time.time()
        Train_auc, y_t_real, y_t_hat = ut.calculate_auc(ehr_model, train_mbs, which_model=wmodel)
        valid_auc, y_v_real, y_v_hat = ut.calculate_auc(ehr_model, valid_mbs, which_model=wmodel)
        
        print(f"Epoch: {ep}, Train_auc: {Train_auc}, Valid_auc: {valid_auc}, Avg Loss: {avg_loss}, Train Time: {train_time}")

        # Save best model
        if valid_auc > bestValidAuc:
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            y_real = y_v_real
            y_hat = y_v_hat
            
        # Early stopping
        if ep - bestValidEpoch > patience: 
            break
            
    return y_real, y_hat

def objective_GRU(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Initialize metrics arrays
    metrics = {
        'auc': np.empty(5),
        'ap': np.empty(5),
        'precision': np.empty(5),
        'recall': np.empty(5),
        'f1': np.empty(5)
    }
    
    print("Starting trial", trial.number)
    
    # Define hyperparameters for this trial
    params = {
        'lr': trial.suggest_categorical('lr_expo', [0.001, 0.005, 0.01, 0.05, 0.1]),
        'l2_expo': trial.suggest_int('l2_expo', -5, -4),
        'eps_expo': trial.suggest_int('eps_expo', -8, -4),
        'embed_dim_expo': trial.suggest_int('embed_dim_expo', 5, 8),
        'hidden_size_expo': trial.suggest_int('hidden_size_expo', 5, 8),
        'optimizer_name': trial.suggest_categorical('optimizer_name', ['Adagrad', 'Adamax'])
    }
    
    # Perform k-fold cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=47)
    X = np.array(train_list)
    
    for fold, (train_index, valid_index) in enumerate(cv.split(X)):
        print(f"Training and Testing for Fold#: {fold+1}")
        print("*****************************************")
        
        # Prepare data for this fold
        X_train, X_valid = X[train_index], X[valid_index] 
        train = EHRDataset(X_train.tolist(), sort=True, model='RNN')
        train_mbs = list(EHRdataloader(train, batch_size=128, packPadMode=True))
        valid = EHRDataset(X_valid.tolist(), sort=True, model='RNN')
        valid_mbs = list(EHRdataloader(valid, batch_size=128, packPadMode=True))
        
        # Initialize model
        ehr_model = model.EHR_RNN(
            input_size_1, 
            embed_dim=2**params['embed_dim_expo'], 
            hidden_size=2**params['hidden_size_expo'], 
            n_layers=2, 
            dropout_r=0.1, 
            cell_type='GRU', 
            bii=True, 
            time=True
        )
        
        if use_cuda: 
            ehr_model = ehr_model.cuda()
        torch.cuda.empty_cache()
        
        # Train model
        RNN_real_label, RNN_predicted_label = run_dl_model(
            ehr_model, 
            train_mbs, 
            valid_mbs,
            wmodel='RNN',
            patience=5,
            l2=10**params['l2_expo'], 
            eps=10**params['eps_expo'], 
            lr=params['lr'],
            opt=params['optimizer_name']
        )
        
        # Calculate metrics
        RNN_predicted_class_label = np.where(np.array(RNN_predicted_label) > 0.5, 1, 0).tolist()
        
        metrics['auc'][fold] = roc_auc_score(RNN_real_label, RNN_predicted_label)
        metrics['ap'][fold] = average_precision_score(RNN_real_label, RNN_predicted_label)
        metrics['precision'][fold], metrics['recall'][fold], metrics['f1'][fold], _ = \
            precision_recall_fscore_support(RNN_real_label, RNN_predicted_class_label, average="binary")
    
    # Log results
    print(f'Mean AUC: {np.mean(metrics["auc"])}, Mean AP: {np.mean(metrics["ap"])}')
    print(f'Mean Precision: {np.mean(metrics["precision"])}, Mean Recall: {np.mean(metrics["recall"])}, Mean F1: {np.mean(metrics["f1"])}')
    
    results_list.append([np.mean(metrics["auc"]), np.mean(metrics["ap"])])
    
    return np.mean(metrics["auc"])

def calculate_class_weights(train_data):
    """Calculate and set global class weights based on training data distribution."""
    global WEIGHT_CLASS_0, WEIGHT_CLASS_1
    
    train_zeros = sum(1 for i in range(len(train_data)) if train_data[i][1] == 0)
    train_ones = sum(1 for i in range(len(train_data)) if train_data[i][1] == 1)
    print(f"Train ratio (ones/zeros): {train_ones/train_zeros}")
    
    total_samples = len(train_data)
    WEIGHT_CLASS_0 = total_samples / (2 * train_zeros)  # Weight for majority class
    WEIGHT_CLASS_1 = total_samples / (2 * train_ones)   # Weight for minority class
    
    print(f"Class weights - Majority (0): {WEIGHT_CLASS_0:.4f}, Minority (1): {WEIGHT_CLASS_1:.4f}")


if __name__ == "__main__":
    # Setup paths and parameters
    start_tm = datetime.now()
    output_folder_name = 'Results_15_final_new'
    days_before_index_date = 1825
    part_common_path = get_common_path('../../../Results_new_no_censored/') + output_folder_name + '/'
    
    print('Running BiGRU optimization')
    selected_list = ['Results_1d_365d_window']
    #selected_list = ['Results_1d_730d_window']
    #selected_list = ['Results_1d_1095d_window']
    #selected_list = ['Results_1d_1825d_window']
    
    
    # Initialize result storage
    DL_result_lists = []
    tuning_results_list = []
    para_value_list = []
    para_name_list = []

    # Process each folder
    for idx_folder, specific_folder_name in enumerate(selected_list):
        common_path = part_common_path + str(days_before_index_date) + "/" + specific_folder_name + '/'        
        print(f'{idx_folder+1}. Running DL for file from {specific_folder_name}')
        
        # Load data
        train_sl, test_sl, valid_sl, types_d = load_input_pkl(common_path)
        types_d_rev = dict(zip(types_d.values(), types_d.keys()))
        
        # Setup model parameters    
        train_list = train_sl + valid_sl  # combine train and validation
        input_size_1 = [len(types_d_rev) + 1]
        
        calculate_class_weights(train_list)
        
        # Run Optuna optimization
        print(f'Starting BiGRU hyperparameter tuning for {specific_folder_name} at {datetime.now()}')
        results_list = []
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_GRU, n_trials=100)
        
        # Log results
        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Trial number: {best_trial.number}")
        print(f"  Value: {best_trial.value}")
        print("  Parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
            para_value_list.append(value)
            if len(para_name_list) < len(best_trial.params):
                para_name_list.append(key)
        
        # Save results
        best_result = results_list[best_trial.number]
        tuning_result_row = [specific_folder_name, 'BiGRU'] + best_result + [best_trial.number, best_trial.params] + para_value_list
        tuning_results_list.append(tuning_result_row)
    
    print(f'Finished all runs at {datetime.now()}, total time: {datetime.now() - start_tm}')
