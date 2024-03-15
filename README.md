# Experience-Based Resource Allocation for Remaining Time Optimization
Code related to the implementation of the paper ''Experience-Based Resource Allocation for Remaining Time Optimization''

Code related to the implementation of the Lambda Function, the Experience Based Oracle Function, and the Profiles generation.

# Installation 

The code has been developed and tested on Linux 22.04 LTS. With a Python 3.10.2 environment. 
The laptop running the code has 16GB of RAM and an Intel Core i7-10750H CPU @ 2.60GHz × 12.
For storing all the Profiles generated, we used the SSD memory of the laptot, with 1TB of space.

The code for testing the recommendations is at https://anonymous.4open.science/r/RecsSysBPSEvaluator-256B.

# Create a python 3.10 venv with the following requirements, also stored in requirements.txt
pandas==1.5.0

pm4py==2.7.8.2

joblib==1.3.2

numpy==1.26.0

scikit-learn==1.3.2

pm4py==2.7.8.2

seaborn==0.13.0 

catboost==1.0.3

ipywidgets==8.1.1

# Usage
Change the hparams value on the top of every script to the desired value. 
The possible hparameter sets are the ones in the folder hparams.

Then, unzip the logs in the "log" folder.

# Contributing
Anonymus

# Licence
None
