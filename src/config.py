import torch
import os
import logging

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
)

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR    = './data_cut_y12/'
RESULTS_DIR = './results/'

# Trainings 
LR_MOE     = 5e-4
EPOCHS_AE  = 40
EPOCHS_MOE = 60
NP         = 49863
WEIGHT_DECAY = 5e-4

# Hyperparameters
TRAIN_AE   = False
BATCH_SIZE = 40_000  
LR_AE      = 4e-3      

# VAE Settings
LATENT_DIM = 3
VAE_BETA = 0.1         
VAE_WARMUP_EPOCHS = 20 
VAE_USE_MU_RECON = False 


# Prior GMM (FIJO)
GMM_PRIOR_CENTERS = [
    [-3.0, 0.0, 0.0, 0.0],
    [ 3.0, 0.0, 0.0, 0.0],
]

VAL_SPLIT = 0.0
