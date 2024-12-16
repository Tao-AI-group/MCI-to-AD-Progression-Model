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
# Predicting Progression from MCI to AD

The progression from Mild Cognitive Impairment (MCI) to Alzheimer’s Disease (AD) is a crucial area of research. Early prediction can enable timely therapeutic interventions and improve patient outcomes. This repository provides a pipeline designed to:

- **Load patient-level longitudinal data:**  
  The code starts with preprocessed pickle files containing patient visits and associated clinical events or measurements.

- **Train deep learning models on time-series data:**  
  Sequential models are implemented to handle the longitudinal nature of the data, capturing temporal patterns and trends in patient health trajectories.

- **Optimize model performance using Optuna:**  
  Hyperparameter tuning is automated with Optuna, ensuring optimal model configurations and improved predictive accuracy.

- **Generate metrics**  
  The pipeline outputs performance metrics, helping users assess and interpret the predictive capabilities of the models.

This repository is designed to accommodate various lookback windows (e.g., 365 days, 730 days, etc.) before the index date. By adjusting these time windows, users can tailor prediction horizons and patient stratification strategies based on disease onset.


## Features
- **Data Handling:** Code for loading, cleaning, and preparing input data (e.g., medical imaging features, demographic information, cognitive assessments).
- **Deep Learning Models:** Implementations of recurrent neural networks and attention-based models tailored for progression prediction.
- **Hyperparameter Tuning:** Automated search for optimal configurations using Optuna.
- **Metrics & Evaluation:** Various metrics (AUC, sensitivity, specificity, etc.) and plotting routines to evaluate model performance.
- **Easy Customization:** Modular code structure, easily adaptable for other longitudinal prediction tasks.

## Repository Structure

