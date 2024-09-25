# Master’s Thesis: **Analysis of Infants' General Movements Using Machine Learning
Techniques for Early Detection of Neurological and Developmental Disorders**

### Author: 
**Maciej Baranowski**

### Written under the supervision of: 
**prof. dr hab. inż Marcin Grzegorzek**

### With additional help and supervision from:
**Dr.-Ing. Frédérica Li**

### Institution: 
**University of Economics in Katowice**

---

## Overview
This repository contains the code and resources developed for my Master's thesis titled **Analysis of Infants' General Movements Using Machine Learning Techniques for Early Detection of Neurological and Developmental Disorders**.
Please note that the data used in the thesis is not directly stored in the repository due to its size and value. For access to the data, it is recommended to contact the author directly. The dataset used in this study was collected as part of the Bundesministerium für Bildung und Forschung (BMBF) project ScreenFM (grant number: 13GW0444). During the data collection process, all legal requirements were strictly adhered to, including obtaining informed consent from the parents of the infants. The data collection and usage were carried out with the full agreement of all involved parties, ensuring that ethical standards were maintained throughout the project.

The repository includes the following:
- **Data preprocessing scripts**: Scripts for handling raw data and preparing it for analysis.
- **Machine learning models**: Implementation of the models used in the thesis, including LSTM, CNN and hybrid LSTM-CNN models.
- **Evaluation scripts**: Scripts to evaluate the models' performance using cross-validation and other methods.
- **Documentation and README**: Instructions on how to set up and run the code.

## Structure of the Repository
- `data/`: Scripts related to data preprocessing and handling.
- `models/`: All the model implementation files.
- `evaluation/`: Scripts to evaluate models, including cross-validation and metrics computation.
- `docs/`: Documentation, including this README and any relevant references.

## Requirements
To reproduce the experiments, you will need the following software and libraries:
- Python 3.x
- Required Python libraries (can be installed via `requirements.txt`):
    - NumPy
    - Pandas
    - PyTorch
    - Scikit-learn
    - Seaborn
    - Matplotlib

```bash
pip install -r requirements.txt
