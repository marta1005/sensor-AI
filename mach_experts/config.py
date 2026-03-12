import torch
import os

# ========= Rutas =========

DATA_DIR = "../data_cut_y12"   # cámbialo si lo tienes en otro sitio

RESULTS_DIR = "./results"
MODEL_DIR   = os.path.join(RESULTS_DIR, "models")
SCALER_DIR  = os.path.join(RESULTS_DIR, "scalers")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# ========= Hardware =========

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Hiperparámetros =========

BATCH_SIZE   = 65536
LR           = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS       = 150   # <<--- SUBIMOS ÉPOCAS

# ========= Estructura de entrada/salida =========

INPUT_DIM  = 9   # [x,y,z,nx,ny,nz,Mach,AoA,Pi]
OUTPUT_DIM = 1   # SOLO Cp

# ========= Regímenes de Mach =========

MACH_SUB_MAX   = 0.65
MACH_TRANS_MIN = 0.65
MACH_TRANS_MAX = 0.85
# M > MACH_TRANS_MAX será supersónico