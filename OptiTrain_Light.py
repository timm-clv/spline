import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PRÉPARATION DES DONNÉES
# ==============================================================================
DATASET_FILE = "dataset_velocity_constrained.npz"

try:
    data = np.load(DATASET_FILE)
    X_np, Y_np = data['X'], data['Y']
    print(f"Chargement réussi : {len(X_np)} échantillons.")
except FileNotFoundError:
    print(f"Erreur : '{DATASET_FILE}' introuvable. Lancez SpGen_Local.py d'abord.")
    exit()

# Normalisation (Essentielle pour la convergence)
# On sauvegarde les stats pour le contrôleur C++ / Python futur
X_mean, X_std = X_np.mean(axis=0), X_np.std(axis=0)
Y_mean, Y_std = Y_np.mean(axis=0), Y_np.std(axis=0)

# Petit fix pour éviter la division par zéro si une colonne est constante (ex: v_start toujours 0)
X_std[X_std < 1e-6] = 1.0 
Y_std[Y_std < 1e-6] = 1.0

np.savez("stats_velocity.npz", X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)
print("Statistiques de normalisation sauvegardées dans 'stats_velocity.npz'.")

# Conversion en Tensors
X_tensor = torch.tensor((X_np - X_mean) / X_std).float()
Y_tensor = torch.tensor((Y_np - Y_mean) / Y_std).float()

# Dataset & Loader
dataset = TensorDataset(X_tensor, Y_tensor)
# Batch size plus grand car le problème est plus simple (moins de sorties)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# ==============================================================================
# 2. ARCHITECTURE DU MODÈLE
# ==============================================================================
class TimePredictorNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Entrée : 6 valeurs (x, y, cos, sin, v_start, v_end)
        # Sortie : 1 valeur (T_optimal)
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )
    
    def forward(self, x):
        return self.net(x)

model = TimePredictorNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ==============================================================================
# 3. ENTRAÎNEMENT
# ==============================================================================
print("Début de l'entraînement...")
EPOCHS = 150
loss_history = []

for epoch in range(EPOCHS):
    total_loss = 0
    for bx, by in loader:
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")

# Sauvegarde
torch.save(model.state_dict(), "time_model.pth")
print("\nModèle sauvegardé : time_model.pth")

# ==============================================================================
# 4. VALIDATION & RECONSTRUCTION (La réponse à votre question)
# ==============================================================================
def reconstruct_control_points(inputs_denorm, T_pred):
    """
    C'est ICI que la magie opère.
    On transforme la sortie unique (T) en courbe complète (P0, P1, P2, P3).
    """
    xf, yf, c_th, s_th, v_s, v_e = inputs_denorm
    
    P0 = np.array([0.0, 0.0])
    P3 = np.array([xf, yf])
    
    # Orientation
    u_start = np.array([1.0, 0.0]) # Robot local
    u_end   = np.array([c_th, s_th]) # Cible locale
    
    # Calcul déterministe de P1 et P2 basé sur T et les Vitesses
    k1 = (v_s * T_pred) / 3.0
    k2 = (v_e * T_pred) / 3.0
    
    P1 = P0 + k1 * u_start
    P2 = P3 - k2 * u_end
    
    return P0, P1, P2, P3

# Test visuel sur un cas aléatoire
model.eval()
with torch.no_grad():
    # Prendre un sample du dataset
    idx = np.random.randint(0, len(X_np))
    sample_in = X_tensor[idx].unsqueeze(0)
    real_out = Y_np[idx]
    real_in_raw = X_np[idx]
    
    # Prédiction
    pred_out_norm = model(sample_in).item()
    T_pred = pred_out_norm * Y_std[0] + Y_mean[0]
    
    print(f"\n--- TEST DE RECONSTRUCTION (Index {idx}) ---")
    print(f"Cible : ({real_in_raw[0]:.2f}, {real_in_raw[1]:.2f})")
    print(f"Vitesses demandées : Départ={real_in_raw[4]:.2f}, Arrivée={real_in_raw[5]:.2f}")
    print(f"Temps Réel (Dataset) : {real_out[0]:.3f}s")
    print(f"Temps Prédit (NN)    : {T_pred:.3f}s")
    
    # Reconstruction Géométrique
    P0, P1, P2, P3 = reconstruct_control_points(real_in_raw, T_pred)
    
    # Génération courbe
    t_vals = np.linspace(0, 1, 50)
    curve = []
    for t in t_vals:
        mu = 1-t
        pt = (mu**3 * P0 + 3*t*mu**2 * P1 + 3*t**2*mu * P2 + t**3 * P3)
        curve.append(pt)
    curve = np.array(curve)
    
    # Affichage
    plt.figure(figsize=(8, 6))
    plt.plot(curve[:,0], curve[:,1], 'b-', linewidth=2, label='Trajectoire Prédite')
    plt.scatter([P0[0], P1[0], P2[0], P3[0]], [P0[1], P1[1], P2[1], P3[1]], c='r')
    plt.plot([P0[0], P1[0]], [P0[1], P1[1]], 'k--', alpha=0.3)
    plt.plot([P2[0], P3[0]], [P2[1], P3[1]], 'k--', alpha=0.3)
    plt.title(f"Reconstruction NN : T={T_pred:.2f}s (vs {real_out[0]:.2f}s)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()