import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# ==============================================================================
# 1. ARCHITECTURE DU MODÈLE (Doit être identique à OptiTrain_Velocity)
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

# ==============================================================================
# 2. CONTRÔLEUR INTELLIGENT
# ==============================================================================
class VelocityController:
    def __init__(self, model_path="time_model.pth", stats_path="stats_velocity.npz"):
        self.device = torch.device('cpu')
        
        # 1. Chargement Architecture & Poids
        self.model = TimePredictorNet().to(self.device)
        try:
            # weights_only=True pour sécurité (si PyTorch récent)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"[OK] Modèle chargé : {model_path}")
        except Exception as e:
            print(f"[ERREUR] Impossible de charger le modèle : {e}")
            raise

        # 2. Chargement Statistiques de Normalisation
        try:
            stats = np.load(stats_path)
            self.X_mean = stats['X_mean']
            self.X_std = stats['X_std']
            self.Y_mean = stats['Y_mean']
            self.Y_std = stats['Y_std']
            print(f"[OK] Stats chargées : {stats_path}")
        except FileNotFoundError:
            print("[ERREUR] Fichier stats introuvable. Avez-vous lancé l'entraînement ?")
            raise

    def global_to_local(self, robot_pos, robot_yaw, target_pos, target_yaw):
        """
        Transforme la cible globale dans le repère du robot.
        """
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        
        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        x_local = c * dx + s * dy
        y_local = -s * dx + c * dy
        
        theta_local = target_yaw - robot_yaw
        theta_local = (theta_local + np.pi) % (2 * np.pi) - np.pi
        
        return x_local, y_local, theta_local

    def compute_trajectory(self, robot_state, target_state, num_points=50):
        """
        Calcule la trajectoire complète (Points de contrôle + Échantillons).
        
        Args:
            robot_state: [x, y, yaw, v_current]
            target_state: [x, y, yaw, v_target]
        
        Returns:
            traj_global (np.array): Points de la courbe (x,y)
            control_points (tuple): (P0, P1, P2, P3) en global
            T_pred (float): Temps estimé pour la manœuvre
        """
        rx, ry, r_yaw, v_start = robot_state
        tx, ty, t_yaw, v_end = target_state
        
        # 1. Passage en Local
        x_loc, y_loc, th_loc = self.global_to_local([rx, ry], r_yaw, [tx, ty], t_yaw)
        
        # 2. Préparation Entrée NN (6 valeurs)
        # [x, y, cos_th, sin_th, v_start, v_end]
        inp_raw = np.array([
            x_loc, y_loc, 
            np.cos(th_loc), np.sin(th_loc), 
            v_start, v_end
        ], dtype=np.float32)
        
        # Normalisation
        inp_norm = (inp_raw - self.X_mean) / (self.X_std + 1e-6)
        inp_tensor = torch.tensor(inp_norm).unsqueeze(0).to(self.device)
        
        # 3. Inférence (Prédiction du TEMPS T)
        with torch.no_grad():
            out_norm = self.model(inp_tensor).item()
        
        # Dénormalisation
        T_pred = out_norm * self.Y_std[0] + self.Y_mean[0]
        
        # Sécurité : T ne peut pas être négatif ou nul
        T_final = max(0.5, T_pred) # On impose au moins 0.5s pour éviter les divisions par zéro
        
        # 4. Reconstruction des Points de Contrôle (LOCAUX)
        # C'est ici qu'on applique la contrainte de vitesse stricte
        P0_loc = np.array([0.0, 0.0])
        P3_loc = np.array([x_loc, y_loc])
        
        # Vecteurs directeurs locaux
        u_start_loc = np.array([1.0, 0.0]) # Robot regarde vers X+ en local
        u_end_loc   = np.array([np.cos(th_loc), np.sin(th_loc)])
        
        # Formule Mathématique Spline
        k1 = (v_start * T_final) / 3.0
        k2 = (v_end * T_final) / 3.0
        
        P1_loc = P0_loc + k1 * u_start_loc
        P2_loc = P3_loc - k2 * u_end_loc
        
        # 5. Transformation vers GLOBAL
        c, s = np.cos(r_yaw), np.sin(r_yaw)
        R = np.array([[c, -s], [s, c]])
        pos_rob = np.array([rx, ry])
        
        def to_global(p_loc): return (R @ p_loc) + pos_rob
        
        P0_glob = to_global(P0_loc)
        P1_glob = to_global(P1_loc)
        P2_glob = to_global(P2_loc)
        P3_glob = to_global(P3_loc)
        
        # 6. Échantillonnage de la courbe (Bézier Cubique)
        t_vals = np.linspace(0, 1, num_points)
        curve_points = []
        for t in t_vals:
            mu = 1 - t
            pt = (mu**3 * P0_glob + 3*t*mu**2 * P1_glob + 3*t**2*mu * P2_glob + t**3 * P3_glob)
            curve_points.append(pt)
            
        return np.array(curve_points), (P0_glob, P1_glob, P2_glob, P3_glob), T_final

# ==============================================================================
# MAIN : DÉMONSTRATION
# ==============================================================================
if __name__ == "__main__":
    # --- CONFIGURATION DU TEST ---
    # Scénario : Le robot arrive vite et doit se garer (vitesse nulle à la fin)
    # Ou : Le robot part de 0 et doit intercepter une cible en mouvement
    
    ROBOT_INIT = [0.0, 0.0, np.radians(0), 1.5]  # x, y, yaw, v_start=1.5 m/s
    TARGET     = [0.0, -2.0, np.radians(90), 4.0] # x, y, yaw, v_end=0.0 m/s (Arrêt)
    
    try:
        controller = VelocityController()
        
        start_time = time.time()
        traj, ctrls, T_opt = controller.compute_trajectory(ROBOT_INIT, TARGET)
        exec_time = (time.time() - start_time) * 1000
        
        print("\n=== RÉSULTAT PLANIFICATION ===")
        print(f"Temps de calcul : {exec_time:.2f} ms")
        print(f"Temps de manœuvre prédit (T) : {T_opt:.2f} s")
        print(f"Vitesse Départ : {ROBOT_INIT[3]} m/s")
        print(f"Vitesse Arrivée: {TARGET[3]} m/s")
        
        # --- VISUALISATION ---
        P0, P1, P2, P3 = ctrls
        
        plt.figure(figsize=(10, 6))
        
        # Trajectoire
        plt.plot(traj[:,0], traj[:,1], 'b-', linewidth=2, label="Trajectoire Optimale")
        
        # Points de contrôle
        plt.plot([P0[0], P1[0]], [P0[1], P1[1]], 'k--', alpha=0.3)
        plt.plot([P2[0], P3[0]], [P2[1], P3[1]], 'k--', alpha=0.3)
        plt.scatter([P0[0], P3[0]], [P0[1], P3[1]], c='g', s=80, zorder=5, label="Départ/Arrivée")
        plt.scatter([P1[0], P2[0]], [P1[1], P2[1]], c='r', s=50, zorder=5, label="Points de Contrôle (Vitesse)")
        
        # Vecteurs Vitesse
        # Vitesse initiale (échelle graphique)
        vec_len_start = ROBOT_INIT[3] * 0.5 
        plt.arrow(P0[0], P0[1], np.cos(ROBOT_INIT[2])*vec_len_start, np.sin(ROBOT_INIT[2])*vec_len_start, 
                  head_width=0.15, color='orange', width=0.05, label=f"V_start ({ROBOT_INIT[3]} m/s)")
        
        # Vitesse finale (si non nulle)
        if TARGET[3] > 0.1:
            vec_len_end = TARGET[3] * 0.5
            plt.arrow(P3[0], P3[1], np.cos(TARGET[2])*vec_len_end, np.sin(TARGET[2])*vec_len_end, 
                      head_width=0.15, color='purple', width=0.05, label=f"V_end ({TARGET[3]} m/s)")
        
        plt.title(f"Planification Cinématique : T={T_opt:.2f}s")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"\n[CRITICAL] Erreur lors de l'exécution : {e}")
        print("Vérifiez que 'time_model.pth' et 'stats_velocity.npz' sont bien générés.")