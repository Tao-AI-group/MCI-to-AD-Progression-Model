# MCI-to-AD Progression Model

**Repository:** [Tao-AI-group/MCI-to-AD-Progression-Model](https://github.com/Tao-AI-group/MCI-to-AD-Progression-Model)

This repository provides code and pipelines for predicting the progression from Mild Cognitive Impairment (MCI) to Alzheimer’s Disease (AD) using deep learning models. The focus is on implementing and fine-tuning recurrent neural network models (particularly BiGRU-based architectures) on longitudinal claims datasets. By leveraging demographic information, and other clinical features, this project aims to identify patterns that can predict conversion from MCI to AD within specific time horizons.

![alt text](https://raw.githubusercontent.com/Tao-AI-group/MCI-to-AD-Progression-Model/main/GA.svg)
## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Contact](#contact)

## Overview

The progression from Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD) is a crucial area of research, as early prediction can facilitate timely therapeutic interventions. This repository houses a pipeline designed to operationalize such predictions by leveraging patient claims data. The pipeline includes the following stages:

1. **Data Preprocessing**:
   - **Age, Gender, Diagnoses, Procedures, and Medications**: Patient claims data are preprocessed where:
     - Diagnoses are mapped to Phecodes.
     - Procedures are converted to Clinical Classification Software (CCS) codes.
     - Medications' National Drug Codes (NDC) are mapped to their generic names.

2. **Construction of Longitudinal Patient Data**:
   - Patient claims are aggregated into 15-day encounter periods leading up to the index date. These longitudinal datasets are then segmented into training, validation, and testing sets, and serialized into `.pkl` files for model training and evaluation.

3. **Model Evaluation**:
   - **BiGRU (Bidirectional Gated Recurrent Unit)**: This deep learning model is specifically designed to address the longitudinal nature of the data, capturing temporal trends in patient health trajectories.
   - **Baseline Models**: A comparative analysis is performed using traditional machine learning models including Light Gradient Boosting Machine (LGBM), XGBoost, Logistic Regression (LR), and Random Forest (RF).

4. **Hyperparameter Optimization with Optuna**:
   - Hyperparameter tuning is automated using Optuna, employing a 5-fold cross-validation scheme to ensure that model configurations are optimized for maximal predictive accuracy.

5. **Performance Metrics Calculation**:
   - A comprehensive suite of performance metrics is generated, enabling users to assess and compare the predictive capabilities of each model.

The code leverages patient data from 5 years prior to the index date to evaluate the model's predictive accuracy of MCI-to-AD progression across multiple time horizons: one, two, three, and five years from the index date.


## Repository Structure

This repository is structured to facilitate access to the various components of the MCI-to-AD progression model. Below is an overview of the directory and file organization:


- **/Preprocessing/**: Contains scripts for preprocessing inputs to the models.
  - **01-Preprocessing_Input.ipynb**: The script is used for preprocess patient claims and create aggregated encounters.
  - **02-Generate_Model_Input.py**: The script generates final model input.

- **/Mapping_Files/**: This directory contains the mapping files used for diagnoses, procedure, and medications .

- **/Models_CV_Optuna/**: Contains the implementation of the BiGRU model and other baseline model architectures including hyperparameter optimization and cross validation code.
  - **BiGRU_model.py**: The script where the BiGRU model is implemented.
    - The input for the BIGRU model is prepared as described in https://github.com/ZhiGroup/pytorch_ehr
  - **LGBM_model.py**: The script where the Light Gradient Boosting Machine model is implemented.
  - **XGBoost_model.py**: The script where the XGBoost model is implemented.
  - **RF_model.py**: The script where the Random Forest model is implemented.
  - **LR_model.py**: The script where the Logistic Regression model is implemented.

- **/Model_Inference_Evaluation/**: Contains scripts for final model training, evaluation and functions for computing various performance metrics.

- **requirements.txt**: A file listing primary Python libraries required by the project.

- **README.md**: Provides an overview and instructions for navigating and utilizing the repository.

## Contact
For questions please contact: Ahmed Abdelhameed abdelhameed.ahmed@mayo.edu


