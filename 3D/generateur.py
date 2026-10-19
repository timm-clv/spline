import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NUM_SAMPLES   = 20000
R_MIN         = 0.5
R_MAX         = 10.0
V_MAX_ROBOT   = 6.5
A_MAX_ROBOT   = 6.0

# ==============================================================================
# 1. MATHÉMATIQUES — BÉZIER CUBIQUE CONTRAINT 3D
# ==============================================================================
def get_control_points_from_velocity(P0, P3, u_start, u_end, v_start, v_end, T):
    # u_start et u_end sont des vecteurs unitaires 3D
    P1 = P0 + (v_start * T / 3.0) * u_start
    P2 = P3 - (v_end   * T / 3.0) * u_end
    return P1, P2

def cubic_bezier_derivatives(t, P0, P1, P2, P3, T_total):
    u = t / T_total
    mu = 1 - u
    dP_du   = 3*mu**2*(P1-P0) + 6*mu*u*(P2-P1) + 3*u**2*(P3-P2)
    d2P_du2 = 6*mu*(P2-2*P1+P0) + 6*u*(P3-2*P2+P1)
    return dP_du / T_total, d2P_du2 / T_total**2

def is_feasible(P0, P3, u_start, u_end, v_start, v_end, T, n_samples=25):
    if T < 0.05:
        return False
    P1, P2 = get_control_points_from_velocity(P0, P3, u_start, u_end, v_start, v_end, T)
    for t in np.linspace(0, T, n_samples):
        vel, acc = cubic_bezier_derivatives(t, P0, P1, P2, P3, T)
        if np.linalg.norm(vel) > V_MAX_ROBOT:
            return False
        if np.linalg.norm(acc) > A_MAX_ROBOT:
            return False
    return True

def find_min_T(P0, P3, u_start, u_end, v_start, v_end,
               T_min=0.05, T_max_init=15.0, tol=0.005, max_iter=40):
    T_high = T_max_init
    for _ in range(15):
        if is_feasible(P0, P3, u_start, u_end, v_start, v_end, T_high): break
        T_high *= 1.5
        if T_high > 200: return None
    else:
        return None

    T_low = T_min
    if is_feasible(P0, P3, u_start, u_end, v_start, v_end, T_low):
        return T_low

    for _ in range(max_iter):
        T_mid = 0.5 * (T_low + T_high)
        if is_feasible(P0, P3, u_start, u_end, v_start, v_end, T_mid):
            T_high = T_mid
        else:
            T_low = T_mid
        if T_high - T_low < tol: break
    return T_high

# ==============================================================================
# 2. GÉNÉRATION DU DATASET
# ==============================================================================
def generate_dataset():
    inputs, outputs = [], []
    P0 = np.array([0.0, 0.0, 0.0])
    u_start = np.array([1.0, 0.0, 0.0]) # Robot pointe vers +X
    rng = np.random.default_rng(seed=42)

    print(f"Génération de {NUM_SAMPLES} trajectoires 3D (360°)...")
    count, skipped = 0, 0

    while count < NUM_SAMPLES:
        # Cible dans une sphère 3D (coordonnées sphériques)
        r = rng.uniform(R_MIN, R_MAX)
        theta = rng.uniform(-np.pi, np.pi)      # Azimut
        phi   = rng.uniform(-np.pi/2, np.pi/2)  # Élévation
        xf = r * np.cos(phi) * np.cos(theta)
        yf = r * np.cos(phi) * np.sin(theta)
        zf = r * np.sin(phi)
        P3 = np.array([xf, yf, zf])

        # Direction cible 3D aléatoire
        t_theta = rng.uniform(-np.pi, np.pi)
        t_phi   = rng.uniform(-np.pi/2, np.pi/2)
        u_end = np.array([
            np.cos(t_phi) * np.cos(t_theta),
            np.cos(t_phi) * np.sin(t_theta),
            np.sin(t_phi)
        ])
        
        # Normalisation au cas où
        u_end /= np.linalg.norm(u_end)

        v_start = rng.uniform(0.0, V_MAX_ROBOT)
        v_end   = rng.uniform(0.0, V_MAX_ROBOT)

        if rng.random() < 0.30: v_start = 0.0
        if rng.random() < 0.30: v_end   = 0.0

        T_valid = find_min_T(P0, P3, u_start, u_end, v_start, v_end)
        if T_valid is None:
            skipped += 1
            continue

        # Inputs : x, y, z, ux, uy, uz, v_start, v_end (8 valeurs)
        inputs.append([xf, yf, zf, u_end[0], u_end[1], u_end[2], v_start, v_end])
        outputs.append([T_valid])
        count += 1
        if count % 2000 == 0:
            print(f"  {count}/{NUM_SAMPLES} générés (rejetés: {skipped})")

    X = np.array(inputs, dtype=np.float32)
    Y = np.array(outputs, dtype=np.float32)
    print(f"\nFini. X:{X.shape}  Y:{Y.shape}")
    return X, Y

if __name__ == "__main__":
    X, Y = generate_dataset()
    np.savez("dataset_generer.npz", X=X, Y=Y)
    print("Dataset sauvegardé : dataset_generer.npz")