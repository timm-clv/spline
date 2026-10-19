"""
generateur.py — Génère un dataset (entrées -> T_opt) pour entraînement NN.
Version POC finale :
  - couverture 360° (cibles devant ET derrière)
  - V_MAX=6.5 m/s, A_MAX=6.0 m/s²
  - recherche de T_opt par bisection (précision 5 ms)
"""
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
# 1. MATHÉMATIQUES — BÉZIER CUBIQUE CONTRAINT
# ==============================================================================
def get_control_points_from_velocity(P0, P3, theta_start, theta_end,
                                     v_start, v_end, T):
    u_start = np.array([np.cos(theta_start), np.sin(theta_start)])
    u_end   = np.array([np.cos(theta_end),   np.sin(theta_end)])
    P1 = P0 + (v_start * T / 3.0) * u_start
    P2 = P3 - (v_end   * T / 3.0) * u_end
    return P1, P2

def cubic_bezier_derivatives(t, P0, P1, P2, P3, T_total):
    u = t / T_total
    mu = 1 - u
    dP_du   = 3*mu**2*(P1-P0) + 6*mu*u*(P2-P1) + 3*u**2*(P3-P2)
    d2P_du2 = 6*mu*(P2-2*P1+P0) + 6*u*(P3-2*P2+P1)
    return dP_du / T_total, d2P_du2 / T_total**2

def is_feasible(P0, P3, theta_start, theta_end, v_start, v_end, T,
                n_samples=25):
    if T < 0.05:
        return False
    P1, P2 = get_control_points_from_velocity(P0, P3, theta_start, theta_end,
                                               v_start, v_end, T)
    for t in np.linspace(0, T, n_samples):
        vel, acc = cubic_bezier_derivatives(t, P0, P1, P2, P3, T)
        if np.sum(vel**2) > V_MAX_ROBOT**2:
            return False
        if np.sum(acc**2) > A_MAX_ROBOT**2:
            return False
    return True

def find_min_T(P0, P3, theta_start, theta_end, v_start, v_end,
               T_min=0.05, T_max_init=15.0, tol=0.005, max_iter=40):
    """Bisection : retourne le T minimal faisable, ou None si infaisable."""
    T_high = T_max_init
    for _ in range(15):
        if is_feasible(P0, P3, theta_start, theta_end, v_start, v_end, T_high):
            break
        T_high *= 1.5
        if T_high > 200:
            return None
    else:
        return None

    T_low = T_min
    if is_feasible(P0, P3, theta_start, theta_end, v_start, v_end, T_low):
        return T_low  # cas dégénéré

    for _ in range(max_iter):
        T_mid = 0.5 * (T_low + T_high)
        if is_feasible(P0, P3, theta_start, theta_end, v_start, v_end, T_mid):
            T_high = T_mid
        else:
            T_low = T_mid
        if T_high - T_low < tol:
            break
    return T_high

# ==============================================================================
# 2. GÉNÉRATION DU DATASET
# ==============================================================================
def generate_dataset():
    inputs, outputs = [], []
    P0 = np.array([0.0, 0.0])
    theta_start = 0.0  # robot orienté +X en local
    rng = np.random.default_rng(seed=42)

    print(f"Génération de {NUM_SAMPLES} trajectoires (360°, "
          f"V_max={V_MAX_ROBOT}, A_max={A_MAX_ROBOT})...")
    count, skipped = 0, 0

    while count < NUM_SAMPLES:
        # Cible en 360°
        r = rng.uniform(R_MIN, R_MAX)
        angle_pos = rng.uniform(-np.pi, np.pi)
        xf, yf = r * np.cos(angle_pos), r * np.sin(angle_pos)
        P3 = np.array([xf, yf])

        theta_end = rng.uniform(-np.pi, np.pi)
        v_start = rng.uniform(0.0, V_MAX_ROBOT)
        v_end   = rng.uniform(0.0, V_MAX_ROBOT)

        # Biais vers les cas fréquents : arrêt au départ/arrivée
        if rng.random() < 0.30: v_start = 0.0
        if rng.random() < 0.30: v_end   = 0.0

        T_valid = find_min_T(P0, P3, theta_start, theta_end,
                              v_start, v_end)
        if T_valid is None:
            skipped += 1
            continue

        inputs.append([xf, yf, np.cos(theta_end), np.sin(theta_end),
                       v_start, v_end])
        outputs.append([T_valid])
        count += 1
        if count % 2000 == 0:
            print(f"  {count}/{NUM_SAMPLES} générés "
                  f"(rejetés: {skipped})")

    X = np.array(inputs, dtype=np.float32)
    Y = np.array(outputs, dtype=np.float32)
    print(f"\nFini. X:{X.shape}  Y:{Y.shape}  (rejetés: {skipped})")
    print(f"T_opt range : [{Y.min():.3f}, {Y.max():.3f}] s  "
          f"mean={Y.mean():.3f} s")
    return X, Y

def visualize_samples(X, Y, num_show=5):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title("Trajectoires (vue du dessus, 360°)")
    ax2.set_title("Profils de vitesse")
    ax2.axhline(V_MAX_ROBOT, color='r', linestyle='--', label='V_max')

    colors = plt.cm.tab10(np.linspace(0, 1, num_show))
    for i in range(num_show):
        xf, yf, c_th, s_th, v_s, v_e = X[i]
        T_opt = Y[i][0]
        P0 = np.array([0, 0])
        P3 = np.array([xf, yf])
        th_end = np.arctan2(s_th, c_th)
        P1, P2 = get_control_points_from_velocity(
            P0, P3, 0.0, th_end, v_s, v_e, T_opt)

        ts = np.linspace(0, T_opt, 80)
        pos, vel = [], []
        for t in ts:
            u = t / T_opt; mu = 1 - u
            p = mu**3*P0 + 3*u*mu**2*P1 + 3*u**2*mu*P2 + u**3*P3
            v, _ = cubic_bezier_derivatives(t, P0, P1, P2, P3, T_opt)
            pos.append(p); vel.append(np.linalg.norm(v))
        pos = np.array(pos)

        c = colors[i]
        ax1.plot(pos[:, 0], pos[:, 1], color=c, linewidth=2,
                 label=f'T={T_opt:.2f}s')
        ax1.scatter(xf, yf, color=c, s=80)
        ax2.plot(ts, vel, color=c, label=f'v0={v_s:.1f},vf={v_e:.1f}')

    ax1.scatter(0, 0, c='g', marker='o', s=150, zorder=5, label='Robot')
    ax1.axis('equal'); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X, Y = generate_dataset()
    np.savez("dataset_generer.npz", X=X, Y=Y)
    print("Dataset sauvegardé : dataset_generer.npz")
    visualize_samples(X, Y, num_show=5)