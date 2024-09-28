### Master’s Thesis: 
**Analysis of Infants' General Movements Using Machine Learning Techniques for Early Detection of Neurological and Developmental Disorders**

### Author: 
**Maciej Baranowski**

### Written under the supervision of: 
**prof. dr hab. inż Marcin Grzegorzek**

### With additional help and supervision from:
**Dr.-Ing. Frédérica Li**

### Institution: 
**University of Economics in Katowice**

------------------------------------------

## Overview
This repository contains the code and resources developed for my Master's thesis titled **Analysis of Infants' General Movements Using Machine Learning Techniques for Early Detection of Neurological and Developmental Disorders**.

## Used data
The dataset utilized in this project was collected as part of the Bundesministerium für Bildung und Forschung (BMBF) project ScreenFM (grant number: 13GW0444). The data collection was conducted by professional medical staff and adheres to the highest ethical and legal standards. Parental consent was obtained for all infants whose movements were analyzed, ensuring full compliance with ethical requirements for the use of human data.

## Data Usage and Access
Please note that due to the sensitive and valuable nature of the data, it is not stored in this repository. For researchers or other parties interested in working with the dataset, direct access can be requested by contacting Prof. Marcin Grzegorzek or Dr.-Ing. Frédérica Li for approval. Only after obtaining proper consent from the data owners will access be granted.

## Cloud usage
The neural networks in this project, due to the large size of the datasets, have been primarily designed to leverage cloud computing platforms like Google Colab and Google Drive. 
To run the code and train the model within a reasonable timeframe, it is highly recommended to use cloud services with GPU support or ensure that your local environment has GPU support enabled.
Training the model using only CPU may result in training times that could last for hours or even longer, depending on the size of the dataset.
While most models also include a script for running on CPU, for performance reasons, it is recommended to use the scripts that are optimized for working with Google Colab and Google Drive. 
If someone wishes to modify the project and run the models on their own GPU, they would need to implement the necessary changes for their local environment.
Please note that some model parameters may slightly differ from those documented in the Master's thesis due to the large number of tests performed during development.

## Repository Content
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

## Technology
To reproduce the experiments, you will need the following software and libraries:
- Python ver. 3.11.3
- Required Python libraries:
    - NumPy ver. 1.26.1
    - Pandas ver. 2.1.2
    - PyTorch ver. 2.4.0+cpu
    - Scikit-learn ver. 1.3.2
    - Seaborn ver. 0.13.0
    - Matplotlib ver. 3.8.0
