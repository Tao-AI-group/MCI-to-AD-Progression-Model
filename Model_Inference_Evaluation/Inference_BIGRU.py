### Tools and Packages
##Basics
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../libs/'))
from datetime import datetime
import math
import pickle 
import os, re

# specify cuda number
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import time
from datetime import datetime
from collections import defaultdict

## ML and Stats

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

## DL Framework
import torch
import torch.nn as nn
from torch import optim

import models as model 
from EHRDataloader import EHRdataloader
from EHRDataloader import EHRdataFromLoadedPickles as EHRDataset
import utils as ut 
import def_function as func

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def load_input_pkl(common_path):
    """Load train, test, validation and type mapping data from pickle files."""
    pkl_result_list = []
    input_name_list = ['.combined.train', '.combined.test', '.combined.valid', '.types']   
    
    for name in input_name_list:
        input_path = common_path + "ad" + name
        pkl_result = pickle.load(open(input_path, 'rb'), encoding='bytes')
        pkl_result_list.append(pkl_result)
        
    return pkl_result_list[0], pkl_result_list[1], pkl_result_list[2], pkl_result_list[3]

def trainbatches(mbs_list, model, optimizer,shuffle = True, loss_fn = nn.BCELoss()):
    
    current_loss = 0
    all_losses =[]
    plot_every = 5
    n_iter = 0 
    
    if shuffle: 
        random.shuffle(mbs_list)

    for i,batch in enumerate(mbs_list):
        sample, label_tensor, seq_l, mtd = batch
        
        #Set Weight
        weight = torch.zeros_like(label_tensor)
        weight[label_tensor==0] = 0.5378951502061653
        weight[label_tensor==1] = 7.097150259067358
    
        loss_fn = nn.BCELoss(weight.squeeze()) 
        
        output, loss = ut.trainsample(sample, label_tensor, seq_l, mtd, model, optimizer, criterion = loss_fn)
        current_loss += loss
        n_iter +=1
    
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
        
    ##Training epochs
    bestValidAuc = 0.0
    bestValidEpoch = 0

    ### epochs loop
    for ep in range(epochs):
        # Training phase
        start = time.time()
        current_loss, train_loss = trainbatches(train_mbs, model = ehr_model, optimizer = optimizer,loss_fn = nn.BCELoss())
        avg_loss = np.mean(train_loss)
        train_time = time_consumption_since(start)

        # Evaluation phase
        eval_start = time.time()
        Train_auc, y_t_real, y_t_hat = ut.calculate_auc(ehr_model, train_mbs, which_model = wmodel)
        valid_auc, y_v_real, y_v_hat = ut.calculate_auc(ehr_model, valid_mbs, which_model = wmodel)
        eval_time = time_consumption_since(eval_start)

        # Log progress
        print(f"Epoch: {ep:3d} | Train AUC: {Train_auc:.4f} | Valid AUC: {valid_auc:.4f} | "
              f"Avg Loss: {avg_loss:.4f} | Train Time: {train_time}")

        # Check for improvement
        if valid_auc > bestValidAuc: 
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            y_real = y_v_real
            y_hat = y_v_hat
            
            # Calculate and display metrics
            y_pred = (np.array(y_hat) > 0.5)
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_real, y_pred))
            print('\nClassification Report:')
            print(classification_report(y_real, y_pred))
            
            # Save best model
            ckpt_dir = os.path.join(results_output_path, 'optuna_tuning', 'objective_BiGRU_MCI', '1d_365d')
            if not os.path.exists(ckpt_dir):
                print(f"Creating checkpoint directory: {ckpt_dir}")
                os.makedirs(ckpt_dir)

            # Save model and parameters
            model_path = os.path.join(ckpt_dir, 'test_model.p')
            param_path = os.path.join(ckpt_dir, 'test_parameters.p')
            torch.save(ehr_model, model_path)
            torch.save(ehr_model.state_dict(), param_path)
            print(f"Saved best model to {model_path}")
        
        # Early stopping check
        if ep - bestValidEpoch > patience: break
         
    # Final results
    bestValidAuc = round(bestValidAuc, 4)
    print(f'Best validation AUC {bestValidAuc:.4f} at epoch {bestValidEpoch}')
    
    return y_real, y_hat


if __name__ == "__main__":
    start_tm = datetime.now()
    
    # Initialize paths and configuration
    output_folder_name = 'Results_15_final_new'
    days_before_index_date = 1825
    part_common_path = get_common_path('../../../Results_new_no_censored/') + output_folder_name + '/'
    results_output_path = func.get_common_path('../../Test_results_no_censored/') + '/' + str(days_before_index_date) + '/'
    
    selected_list = ['Results_1d_365d_window']

    # Initialize result tracking
    tuning_results_list = []
    para_value_list = []
    para_name_list = []

    for idx_folder, specific_folder_name in enumerate(selected_list):
        common_path = part_common_path + str(days_before_index_date) + "/" + specific_folder_name + '/'        
        print(f'{idx_folder+1}. Running DL for file from {specific_folder_name}:', '\n')  
         
        
        # Load and prepare data
        train_sl, test_sl, valid_sl, types_d = load_input_pkl(common_path)
        types_d_rev = dict(zip(types_d.values(), types_d.keys()))
        input_size_1 = [len(types_d_rev) + 1]
        
        # Combine train and validation sets
        train_list = train_sl + valid_sl
                       
        to_infere_BiGRU_results_label = 0
        if  to_infere_BiGRU_results_label:
            
            #Fit the best model on the train+validation data and save the model in the test_results folder
            #create the .csv file to generate ['classifiers', 'fpr','tpr','auc', 'auprc']
            
            # Define optimal parameters
            params = {
                'lr': 0.05,
                'l2_expo': -4,
                'eps_expo': -7,
                'embed_dim_expo': 8,
                'hidden_size_expo': 5,
                'optimizer_name': 'Adagrad'
            }
            
            # Initialize and train model
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

            # Prepare data loaders - keeping original format
            train = EHRDataset(train_list, sort=True, model='RNN')
            train_mbs = list(EHRdataloader(train, batch_size=128, packPadMode=True))
            
            test = EHRDataset(test_sl, sort=True, model='RNN')
            test_mbs = list(EHRdataloader(test, batch_size=128, packPadMode=True))

            # Train and evaluate
            RNN_real_label, RNN_predicted_label = run_dl_model(
                ehr_model,
                train_mbs,
                test_mbs,
                wmodel='RNN',
                patience=5,
                l2=10**params['l2_expo'],
                eps=10**params['eps_expo'],
                lr=params['lr'],
                opt=params['optimizer_name']
            )

            # Process predictions
            RNN_predicted_class_label = np.where(np.array(RNN_predicted_label) > 0.5, 1, 0).tolist()
            
            print('\n Classification Report:', classification_report(RNN_real_label, RNN_predicted_class_label))
        
            # Print classification report
            print('\n Classification Report:', classification_report(RNN_real_label, RNN_predicted_class_label))
            
            # Calculate metrics
            auc = roc_auc_score(RNN_real_label, RNN_predicted_label)
            auprc = average_precision_score(RNN_real_label, RNN_predicted_label)
            precision, recall, fscore, _ = precision_recall_fscore_support(RNN_real_label, RNN_predicted_class_label, average='binary')
            precision_weighted, recall_weighted, fscore_weighted, _ = precision_recall_fscore_support(RNN_real_label, RNN_predicted_class_label, average='weighted')
            
            # Calculate confusion matrix metrics
            tn, fp, fn, tp = confusion_matrix(RNN_real_label, RNN_predicted_class_label).ravel()
            specificity = tn / (tn + fp)

            # Save results
            result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc', 'auprc' ])
            fpr, tpr, ths = m.roc_curve(RNN_real_label, RNN_predicted_label)
            
            
            result_table = result_table.append({
                'classifiers': "BiGRU",
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': auc,
                'auprc': auprc
            }, ignore_index=True)
            
            result_table.set_index('classifiers', inplace=True)
            result_table.to_csv(results_output_path + "result_table_MCI_1d_365d.csv", index=False)
            
        # Bootstrap performance evaluation    
        to_calculate_bootstrapped_performance = 1
        if to_calculate_bootstrapped_performance:
            
            # Load best model
            ckpt_dir = os.path.join(results_output_path, 'optuna_tuning/objective_BiGRU_MCI/1d_365d')
            best_model = torch.load(os.path.join(ckpt_dir, "test_model.p"))
            best_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "test_parameters.p")))
            
            if use_cuda:
                best_model.cuda()
            best_model.eval()
            
            # Prepare test data - keeping original format
            test2 = EHRDataset(test_sl, sort=True, model='RNN')
            test_mbs2 = list(EHRdataloader(test2, batch_size=128, packPadMode=True))

            # Calculate performance metrics
            TestAuc, labels_t, y_score = ut.calculate_auc(best_model, test_mbs2, which_model='RNN')
            y_pred = np.where(np.array(y_score) > 0.5, 1, 0).tolist()
            
            # Calculate and save PR curve data
            precision, recall, _ = m.precision_recall_curve(labels_t, y_score)

            with open(os.path.join(results_output_path, 'precision_BiGRU_1d_365d.pkl'), 'wb') as f:
                pickle.dump(precision.tolist(), f)
            with open(os.path.join(results_output_path, 'recall_BiGRU_1d_365d.pkl'), 'wb') as f:
                pickle.dump(recall.tolist(), f)
            
            
            # Perform bootstrap analysis
            n_trials = 500
            alpha = 0.05
            all_CI = defaultdict(list)
            
            print("Starting bootstrap trials...")
            for trial_num in range(n_trials):
                test_resample = resample(test_sl)
                test = EHRDataset(test_resample, sort=True, model='RNN')
                test_mbs = list(EHRdataloader(test, batch_size=128, packPadMode=True))
                
                TestAuc, y_real, y_hat = ut.calculate_auc(best_model, test_mbs, which_model='RNN')
                y_pred = np.where(np.array(y_hat) > 0.5, 1, 0).tolist()
                
                # Calculate metrics for this trial
                precision, recall, fscore, _ = precision_recall_fscore_support(y_real, y_pred, average='weighted')
                tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
                specificity = tn / (tn + fp)
                
                # Store metrics
                metrics = {
                    "auc": roc_auc_score(y_real, y_hat),
                    "auprc": average_precision_score(y_real, y_hat),
                    "precision": precision_score(y_real, y_pred),
                    "recall": recall_score(y_real, y_pred),
                    "f1": f1_score(y_real, y_pred),
                    "precision_weighted": precision,
                    "recall_weighted": recall,
                    "f1_weighted": fscore,
                    "accuracy": accuracy_score(y_real, y_pred),
                    "balanced accuracy": balanced_accuracy_score(y_real, y_pred),
                    "specificity": specificity
                }
                
                for key, value in metrics.items():
                    all_CI[key].append(value)
                
                print(f"Trial number {trial_num} finished")
            
            # Calculate confidence intervals
            df = pd.DataFrame(columns=["lower", "upper", "95%_CI", "stat", "model"])
            
            for key in all_CI.keys():
                p = (alpha/2) * 100
                lower = round(max(0, np.percentile(all_CI[key], p)), 4)
                p = (1-alpha/2) * 100
                upper = round(min(1, np.percentile(all_CI[key], p)), 4)
                
                df.loc[len(df)] = [
                    lower,
                    upper,
                    f"({lower},{upper})",
                    key,
                    os.path.splitext(os.path.basename('1d_365d'))[0]
                ]
            
            df.to_csv(os.path.join(results_output_path, "confidence_interval_BiGRU_MCI_1d_365d.csv"), index=False)  
            
        print(f'Finished running all models, total time: {datetime.now() - start_tm}')
        print('Optimization complete!')
