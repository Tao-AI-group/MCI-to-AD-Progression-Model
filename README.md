# MCI-to-AD Progression Model

**Repository:** [Tao-AI-group/MCI-to-AD-Progression-Model](https://github.com/Tao-AI-group/MCI-to-AD-Progression-Model)

This repository provides code and pipelines for predicting the progression from Mild Cognitive Impairment (MCI) to Alzheimer’s Disease (AD) using deep learning models. The focus is on implementing and fine-tuning recurrent neural network models (particularly BiGRU-based architectures) on longitudinal clinical datasets. By leveraging event sequences, demographic information, and other clinical features, this project aims to identify patterns that can predict conversion from MCI to AD within specific time horizons.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Customization](#customization)
- [Citing This Work](#citing-this-work)
- [License](#license)
- [Contact](#contact)

## Overview
The progression from MCI to AD is a crucial area of research as early prediction can enable timely therapeutic interventions. This repository operationalizes a pipeline that:

1. **Preprocesses patient claims data including**:  Age, Gender, Diagnoses, Procedures and Medications
   Diagnoses are mapped into Phecodes. Procedures are converted to CCS (Clinical Classification Software) Codes. Medications NDC are mapped to generic Names.     
   
2. **Constructs patient-level longitudinal data:**
   Claims data are grouped and converted into encounters every 15 days before the index date. Model input is processed and converted into train/validation/test datasets and saved in .pkl files

3. **Loads patient-level longitudinal data** from preprocessed pickle files. 

4. **Train BiGRU deep learning and four machine learning models on time-series data:**  
  BiGRU sequential model is implemented to handle the longitudinal nature of the data, capturing temporal patterns and trends in patient health trajectories.

5. **Optimize model performance using Optuna:**  
  Hyperparameter tuning is automated with Optuna, using 5-fold cross validation ensuring optimal model configurations and improved predictive accuracy.

6. **Generate metrics**  
  The pipeline outputs performance metrics, helping users assess and interpret the predictive capabilities of the models.

The code is designed to integrate different lookback windows (e.g., 365 days, 730 days, etc.) before the index date to tailor prediction horizons and patient stratification based on disease onset. By adjusting these time windows, users can tailor prediction horizons and patient stratification strategies based on disease onset.


## Features
- **Data Handling:** Code for loading, cleaning, and preparing input data (e.g., medical imaging features, demographic information, cognitive assessments).
- **Deep Learning Models:** Implementations of recurrent neural networks and attention-based models tailored for progression prediction.
- **Hyperparameter Tuning:** Automated search for optimal configurations using Optuna.
- **Metrics & Evaluation:** Various metrics (AUC, sensitivity, specificity, etc.) and plotting routines to evaluate model performance.
- **Easy Customization:** Modular code structure, easily adaptable for other longitudinal prediction tasks.

## Repository Structure

