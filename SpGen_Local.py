import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NUM_SAMPLES = 10000      # Dataset
R_MAX = 10.0             # Distance max de la cible
V_MAX_ROBOT = 2.0        # Vitesse max robot (m/s)
A_MAX_ROBOT = 1.5        # Accélération max tolérée (m/s^2)
DT_SIM = 0.05            # Pas de temps pour l'intégration

# ==============================================================================
# 1. MATHS : BÉZIER CUBIQUE CONTRAINT PAR LA VITESSE
# ==============================================================================

def get_control_points_from_velocity(P0, P3, theta_start, theta_end, v_start, v_end, T):
    """
    Calcule P1 et P2 pour respecter EXACTEMENT v_start et v_end sur une durée T.
    Mathématiques: V(0) = 3(P1-P0)/T  =>  P1 = P0 + (v_start * T / 3) * u_start
    """
    # Vecteurs unitaires
    u_start = np.array([np.cos(theta_start), np.sin(theta_start)])
    u_end   = np.array([np.cos(theta_end), np.sin(theta_end)])
    
    # Calcul des points de contrôle contraints
    # Si T est très petit, les points se rapprochent (risque d'accélération infinie géré plus loin)
    dist_tan_start = (v_start * T) / 3.0
    dist_tan_end   = (v_end * T) / 3.0
    
    P1 = P0 + dist_tan_start * u_start
    P2 = P3 - dist_tan_end * u_end
    
    return P1, P2

def cubic_bezier_derivatives(t, P0, P1, P2, P3, T_total):
    """ Retourne accélération et vitesse au temps t """
    u = t / T_total
    mu = 1 - u
    
    # Vitesse (Dérivée 1ère)
    # dP/du = 3(1-u)^2(P1-P0) + 6(1-u)u(P2-P1) + 3u^2(P3-P2)
    # V = (dP/du) / T
    dPos_du = 3 * mu**2 * (P1 - P0) + 6 * mu * u * (P2 - P1) + 3 * u**2 * (P3 - P2)
    vel = dPos_du / T_total
    
    # Accélération (Dérivée 2nde)
    d2Pos_du2 = 6 * mu * (P2 - 2*P1 + P0) + 6 * u * (P3 - 2*P2 + P1)
    acc = d2Pos_du2 / (T_total**2)
    
    return vel, acc

def evaluate_trajectory_feasibility(P0, P3, theta_start, theta_end, v_start, v_end, T):
    """
    Vérifie si la trajectoire générée par ce T respecte les limites physiques.
    """
    if T < 0.2: return False, np.inf # Temps trop court = accélération infinie
    
    P1, P2 = get_control_points_from_velocity(P0, P3, theta_start, theta_end, v_start, v_end, T)
    
    # Échantillonnage pour vérifier V_max et A_max
    # On vérifie 10 points clés pour aller vite
    t_vals = np.linspace(0, T, 15)
    
    max_v_sq = 0.0
    max_a_sq = 0.0
    
    for t in t_vals:
        vel, acc = cubic_bezier_derivatives(t, P0, P1, P2, P3, T)
        v_sq = np.sum(vel**2)
        a_sq = np.sum(acc**2)
        
        if v_sq > max_v_sq: max_v_sq = v_sq
        if a_sq > max_a_sq: max_a_sq = a_sq
    
    # Vérification stricte
    if max_v_sq > (V_MAX_ROBOT * 1.1)**2: return False, max_v_sq # Tolérance 10%
    if max_a_sq > (A_MAX_ROBOT * 1.1)**2: return False, max_a_sq
    
    return True, 0.0

# ==============================================================================
# 2. GÉNÉRATION INTELLIGENTE (SOLVER)
# ==============================================================================

def generate_dataset_constrained():
    """
    Génère un dataset (Entrées) -> (Sortie: T_optimal)
    Les entrées incluent désormais v_start et v_end.
    """
    inputs = []  # [xf, yf, cos_th, sin_th, v_start, v_end]
    outputs = [] # [T_optimal] (P1 et P2 sont déduits, pas appris)
    
    P0 = np.array([0.0, 0.0]) # Robot toujours à l'origine en local
    theta_start = 0.0         # Robot toujours orienté vers X+ en local
    
    print(f"Génération de {NUM_SAMPLES} trajectoires contraintes...")
    rng = np.random.default_rng()
    
    count = 0
    while count < NUM_SAMPLES:
        # 1. Situation Géométrique Aléatoire
        r = rng.uniform(0.5, R_MAX)
        angle_pos = rng.uniform(-np.pi/2, np.pi/2) # Cible devant
        xf, yf = r * np.cos(angle_pos), r * np.sin(angle_pos)
        P3 = np.array([xf, yf])
        
        theta_end = rng.uniform(-np.pi, np.pi)
        
        # 2. Vitesses Aléatoires (C'est nouveau !)
        # On entraine le modèle à gérer des départs lancés et des arrivées rapides ou lentes
        v_start = rng.uniform(0.0, V_MAX_ROBOT)
        v_end   = rng.uniform(0.0, V_MAX_ROBOT)
        
        # Cas particulier : Arrivée arrêtée (fréquent)
        if rng.random() < 0.3: v_end = 0.0
        # Cas particulier : Départ arrêté
        if rng.random() < 0.3: v_start = 0.0

        # 3. Recherche du Temps Optimal (Dichotomie ou balayage)
        # On cherche le T le plus petit possible qui respecte les limites
        T_valid = None
        
        # On teste des temps de plus en plus longs. 
        # Si T est trop court, l'accélération explose.
        # Si T est trop long, c'est mou mais valide. On veut le plus court valide.
        for T_test in np.linspace(0.5, 8.0, 30):
            is_valid, _ = evaluate_trajectory_feasibility(P0, P3, theta_start, theta_end, v_start, v_end, T_test)
            if is_valid:
                T_valid = T_test
                break # On a trouvé le temps minimal réalisable (Time Optimal)
        
        if T_valid is not None:
            # Sauvegarde
            # INPUT: 6 valeurs (Géométrie + Cinématique)
            inputs.append([xf, yf, np.cos(theta_end), np.sin(theta_end), v_start, v_end])
            # OUTPUT: 1 valeur (Le temps nécessaire)
            outputs.append([T_valid])
            
            count += 1
            if count % 1000 == 0: print(f"  {count}/{NUM_SAMPLES} générés.")
            
    return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)

# ==============================================================================
# 3. VISUALISATION (Inspiré de tanminhcode.py)
# ==============================================================================
def visualize_samples(X, Y, num_show=3):
    plt.figure(figsize=(14, 5))
    
    # Sous-graphe 1 : Trajectoires XY
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Trajectoires (Vue Dessus)")
    
    # Sous-graphe 2 : Profils de Vitesse
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Profils de Vitesse")
    ax2.axhline(V_MAX_ROBOT, color='r', linestyle='--', label='V_max')
    
    colors = ['b', 'g', 'm', 'c']
    
    for i in range(num_show):
        # Récupération données
        xf, yf, c_th, s_th, v_s, v_e = X[i]
        T_opt = Y[i][0]
        
        # Reconstruction
        P0 = np.array([0, 0])
        P3 = np.array([xf, yf])
        th_end = np.arctan2(s_th, c_th)
        
        P1, P2 = get_control_points_from_velocity(P0, P3, 0, th_end, v_s, v_e, T_opt)
        
        # Génération points pour plot
        ts = np.linspace(0, T_opt, 50)
        pos_arr = []
        vel_arr = []
        
        for t in ts:
            p, _, _ = cubic_bezier(t, P0, P1, P2, P3, T_opt) # Réutilisation fct helper classique
            v_vec, _ = cubic_bezier_derivatives(t, P0, P1, P2, P3, T_opt)
            pos_arr.append(p)
            vel_arr.append(np.linalg.norm(v_vec))
            
        pos_arr = np.array(pos_arr)
        
        # Plot XY
        c = colors[i % len(colors)]
        ax1.plot(pos_arr[:,0], pos_arr[:,1], color=c, label=f'Traj {i}')
        ax1.arrow(0, 0, 0.5, 0, color=c, width=0.05) # Départ
        ax1.scatter(xf, yf, color=c)
        
        # Plot Vitesse
        ax2.plot(ts, vel_arr, color=c, label=f'v0={v_s:.1f}, vf={v_e:.1f}')
        
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()
    ax2.grid(True)
    ax2.legend()
    plt.show()

# Helper pour visualisation (copie simplifiée de la logique bezier standard)
def cubic_bezier(t, P0, P1, P2, P3, T):
    u = t/T; mu=1-u
    return (mu**3*P0 + 3*u*mu**2*P1 + 3*u**2*mu*P2 + u**3*P3), 0, 0

if __name__ == "__main__":
    X_data, Y_data = generate_dataset_constrained()
    
    np.savez("dataset_velocity_constrained.npz", X=X_data, Y=Y_data)
    print("Dataset généré : dataset_velocity_constrained.npz")
    print(f"Input Shape: {X_data.shape} (x, y, cos, sin, v_start, v_end)")
    print(f"Output Shape: {Y_data.shape} (T_optimal)")
    
    visualize_samples(X_data, Y_data, num_show=5)