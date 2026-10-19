"""
entrainement.py — Entraîne le NN TimePredictorNet (6->18->5->1) sur
le dataset 360°. Split train/val 80/20, sauvegarde le meilleur modèle.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DONNÉES
# ==============================================================================
DATASET_FILE = "dataset_generer.npz"
data = np.load(DATASET_FILE)
X_np, Y_np = data['X'], data['Y']
print(f"Chargement : {len(X_np)} échantillons")

# Split 80/20
rng = np.random.default_rng(seed=0)
perm = rng.permutation(len(X_np))
n_train = int(0.8 * len(X_np))
train_idx, val_idx = perm[:n_train], perm[n_train:]

X_tr, Y_tr = X_np[train_idx], Y_np[train_idx]
X_va, Y_va = X_np[val_idx],   Y_np[val_idx]

# Normalisation : stats sur TRAIN uniquement (pas de fuite de val)
X_mean, X_std = X_tr.mean(axis=0), X_tr.std(axis=0)
Y_mean, Y_std = Y_tr.mean(axis=0), Y_tr.std(axis=0)
X_std[X_std < 1e-6] = 1.0
Y_std[Y_std < 1e-6] = 1.0
np.savez("statistique_de_normalisation.npz",
         X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)
print(f"Stats sauvegardées. X_mean={X_mean.round(2)}  Y_mean={Y_mean.round(2)}")

X_tr_t = torch.tensor((X_tr - X_mean) / X_std).float()
Y_tr_t = torch.tensor((Y_tr - Y_mean) / Y_std).float()
X_va_t = torch.tensor((X_va - X_mean) / X_std).float()
Y_va_t = torch.tensor((Y_va - Y_mean) / Y_std).float()

train_loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t),
                          batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_va_t, Y_va_t),
                          batch_size=256, shuffle=False)

# ==============================================================================
# 2. MODÈLE
# ==============================================================================
class TimePredictorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 18), nn.ReLU(),
            nn.Linear(18, 5), nn.ReLU(),
            nn.Linear(5, 1),
        )
    def forward(self, x):
        return self.net(x)

model = TimePredictorNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=10, min_lr=1e-5)
criterion = nn.MSELoss()

# ==============================================================================
# 3. BOUCLE D'ENTRAÎNEMENT
# ==============================================================================
EPOCHS = 300
best_val = float('inf')
best_epoch = 0
train_hist, val_hist = [], []

print("Entraînement...")
for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    tr_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(bx)
    tr_loss /= len(train_loader.dataset)

    # --- Val ---
    model.eval()
    with torch.no_grad():
        va_loss = 0.0
        for bx, by in val_loader:
            va_loss += criterion(model(bx), by).item() * len(bx)
        va_loss /= len(val_loader.dataset)

    scheduler.step(va_loss)
    train_hist.append(tr_loss)
    val_hist.append(va_loss)

    if va_loss < best_val:
        best_val = va_loss
        best_epoch = epoch
        torch.save(model.state_dict(), "modele_temps.pth")

    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | train={tr_loss:.5f} "
              f"val={va_loss:.5f} | best={best_val:.5f} "
              f"@{best_epoch} | lr={lr_now:.1e}")

print(f"\nBest model : epoch {best_epoch}, val loss = {best_val:.5f}")
print("Sauvegardé : modele_temps.pth")

# ==============================================================================
# 4. ÉVALUATION FINALE
# ==============================================================================
model.load_state_dict(torch.load("modele_temps.pth", weights_only=True))
model.eval()
with torch.no_grad():
    pred_norm = model(X_va_t).numpy().flatten()
pred = pred_norm * Y_std[0] + Y_mean[0]
true = Y_va.flatten()
err = pred - true
print(f"\n--- VALIDATION FINALE (sur {len(true)} échantillons) ---")
print(f"MAE  : {np.abs(err).mean():.4f} s")
print(f"RMSE : {np.sqrt((err**2).mean()):.4f} s")
print(f"Max  : {np.abs(err).max():.4f} s")
print(f"MAE relative : {(np.abs(err)/true).mean()*100:.2f} %")

# Courbe de loss
plt.figure(figsize=(9, 5))
plt.plot(train_hist, label='train')
plt.plot(val_hist,   label='val')
plt.axvline(best_epoch, color='r', linestyle='--',
            label=f'best epoch ({best_epoch})')
plt.yscale('log')
plt.xlabel('Epoch'); plt.ylabel('MSE (normalisé)')
plt.title('Courbe de loss')
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.show()