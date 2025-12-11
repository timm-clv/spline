import torch
import torch.nn as nn
import numpy as np

# --- 1. Définition de l'Architecture (Celle de OptiTrain_Light.py) ---
class TimePredictorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

# --- 2. Chargement ---
model = TimePredictorNet()
try:
    model.load_state_dict(torch.load("time_model.pth", weights_only=True))
    model.eval()
    print("Modèle chargé.")
except Exception as e:
    print(f"Erreur chargement modèle: {e}")
    exit()

# Chargement stats
stats = np.load("stats_velocity.npz")
X_mean, X_std = stats['X_mean'], stats['X_std']
Y_mean, Y_std = stats['Y_mean'], stats['Y_std']

# --- 3. Export vers Format Texte Simple pour C++ ---
# Format : 
# HEADER: X_mean(6) X_std(6) Y_mean(1) Y_std(1)
# LAYER: Rows Cols
# WEIGHTS...
# BIAS...
with open("data/model_weights.txt", "w") as f:
    # A. Stats Normalisation
    f.write("STATS\n")
    f.write(" ".join(map(str, X_mean)) + "\n")
    f.write(" ".join(map(str, X_std)) + "\n")
    f.write(f"{Y_mean[0]} {Y_std[0]}\n")
    
    # B. Poids des couches
    # On itère sur les modules Linear
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            f.write("LAYER\n")
            # PyTorch stocke W en [Out, In], nous écrirons tel quel
            rows, cols = layer.weight.shape
            f.write(f"{rows} {cols}\n")
            
            # Écriture Poids (W)
            w_flat = layer.weight.detach().numpy().flatten()
            f.write(" ".join(map(str, w_flat)) + "\n")
            
            # Écriture Biais (b)
            b_flat = layer.bias.detach().numpy().flatten()
            f.write(" ".join(map(str, b_flat)) + "\n")

print("Export terminé vers data/model_weights.txt")