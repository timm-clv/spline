import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# ==============================================================================
# CONFIGURATION ROBOT
# ==============================================================================
# Ajustez ces valeurs selon votre robot réel
V_MAX_ROBOT = 3.5   # m/s
A_MAX_ROBOT = 4   # m/s^2

# ==============================================================================
# 1. ARCHITECTURE DU MODÈLE (Identique à l'entraînement)
# ==============================================================================
class TimePredictorNet(nn.Module):
    def __init__(self):
        super().__init__()
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
# 2. CONTRÔLEUR INTELLIGENT AVEC OPTIMISEUR PHYSIQUE
# ==============================================================================
class VelocityController:
    def __init__(self, model_path="time_model.pth", stats_path="stats_velocity.npz"):
        self.device = torch.device('cpu')
        
        # Chargement Modèle
        self.model = TimePredictorNet().to(self.device)
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"[INIT] Modèle chargé : {model_path}")
        except Exception as e:
            print(f"[ERREUR] Impossible de charger le modèle : {e}")
            raise

        # Chargement Stats
        try:
            stats = np.load(stats_path)
            self.X_mean, self.X_std = stats['X_mean'], stats['X_std']
            self.Y_mean, self.Y_std = stats['Y_mean'], stats['Y_std']
            print(f"[INIT] Stats chargées.")
        except FileNotFoundError:
            raise FileNotFoundError("Fichier stats_velocity.npz manquant.")

    def global_to_local(self, robot_pos, robot_yaw, target_pos, target_yaw):
        dx, dy = target_pos[0] - robot_pos[0], target_pos[1] - robot_pos[1]
        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        x_loc = c * dx + s * dy
        y_loc = -s * dx + c * dy
        theta_loc = (target_yaw - robot_yaw + np.pi) % (2 * np.pi) - np.pi
        return x_loc, y_loc, theta_loc

    def get_bezier_derivatives(self, t, P0, P1, P2, P3, T_total):
        """ Retourne vecteurs vitesse et accélération au temps t """
        u = t / T_total
        mu = 1 - u
        
        # Dérivées par rapport à u (géométrique)
        # P'(u)
        dP_du = 3 * mu**2 * (P1 - P0) + 6 * mu * u * (P2 - P1) + 3 * u**2 * (P3 - P2)
        # P''(u)
        d2P_du2 = 6 * mu * (P2 - 2*P1 + P0) + 6 * u * (P3 - 2*P2 + P1)
        
        # Conversion temporel (Chain rule)
        # v(t) = P'(u) * (1/T)
        vel = dP_du * (1.0 / T_total)
        # a(t) = P''(u) * (1/T^2)
        acc = d2P_du2 * (1.0 / T_total**2)
        
        return vel, acc

    def refine_time_with_physics(self, P0, P3, u_start, u_end, v_start, v_end, T_init):
        """
        Algorithme d'optimisation (Post-Processing).
        Ajuste T pour que max(velocity) <= V_MAX et max(accel) <= A_MAX
        tout en étant le plus rapide possible.
        """
        T_curr = T_init
        
        # Boucle d'optimisation (Max 10 itérations)
        for i in range(10):
            # 1. Calcul des points de contrôle pour ce T
            k1 = (v_start * T_curr) / 3.0
            k2 = (v_end * T_curr) / 3.0
            P1 = P0 + k1 * u_start
            P2 = P3 - k2 * u_end
            
            # 2. Audit de la courbe (Sampling)
            # On vérifie 15 points pour trouver les pics
            t_vals = np.linspace(0, T_curr, 15)
            max_v_sq = 0.0
            max_a_sq = 0.0
            
            for t in t_vals:
                vel, acc = self.get_bezier_derivatives(t, P0, P1, P2, P3, T_curr)
                v_sq = np.sum(vel**2)
                a_sq = np.sum(acc**2)
                if v_sq > max_v_sq: max_v_sq = v_sq
                if a_sq > max_a_sq: max_a_sq = a_sq
            
            max_v = np.sqrt(max_v_sq)
            max_a = np.sqrt(max_a_sq)
            
            # 3. Calcul du ratio de correction
            # On veut : max_v / ratio_v = V_MAX  => ratio_v = max_v / V_MAX
            # On veut : max_a / ratio_a^2 = A_MAX => ratio_a = sqrt(max_a / A_MAX)
            # (Car v proportionnel à 1/T, a proportionnel à 1/T^2)
            
            ratio_v = max_v / (V_MAX_ROBOT * 0.95) # Marge de sécu 5%
            ratio_a = np.sqrt(max_a / (A_MAX_ROBOT * 0.95))
            
            correction_factor = max(ratio_v, ratio_a)
            
            # Si on est proche de l'optimal (entre 95% et 100% des capacités), on arrête
            if 0.95 <= correction_factor <= 1.05:
                break
                
            # Mise à jour de T
            # Si factor > 1 (Violations) -> On augmente T -> On ralentit
            # Si factor < 1 (Sous-régime) -> On diminue T -> On accélère
            T_new = T_curr * correction_factor
            
            # Sécurité anti-boucle infinie ou T trop petit
            if abs(T_new - T_curr) < 0.01: break
            T_curr = max(0.2, T_new)

        return T_curr

    def compute_trajectory(self, robot_state, target_state, num_points=100):
        rx, ry, r_yaw, v_start = robot_state
        tx, ty, t_yaw, v_end = target_state
        
        # 1. Prédiction NN (Initial Guess)
        x_loc, y_loc, th_loc = self.global_to_local([rx, ry], r_yaw, [tx, ty], t_yaw)
        inp = np.array([x_loc, y_loc, np.cos(th_loc), np.sin(th_loc), v_start, v_end], dtype=np.float32)
        inp_norm = (inp - self.X_mean) / (self.X_std + 1e-6)
        
        with torch.no_grad():
            T_pred_norm = self.model(torch.tensor(inp_norm).unsqueeze(0).to(self.device)).item()
        
        T_init = max(0.5, T_pred_norm * self.Y_std[0] + self.Y_mean[0])
        
        # 2. Préparation pour Optimisation (En Local)
        P0_loc = np.array([0., 0.])
        P3_loc = np.array([x_loc, y_loc])
        u_start_loc = np.array([1., 0.])
        u_end_loc = np.array([np.cos(th_loc), np.sin(th_loc)])
        
        # 3. Optimisation Physique (Ajustement de T)
        T_opt = self.refine_time_with_physics(P0_loc, P3_loc, u_start_loc, u_end_loc, v_start, v_end, T_init)
        
        # 4. Génération Finale
        k1 = (v_start * T_opt) / 3.0
        k2 = (v_end * T_opt) / 3.0
        P1_loc = P0_loc + k1 * u_start_loc
        P2_loc = P3_loc - k2 * u_end_loc
        
        # Transfo Global
        c, s = np.cos(r_yaw), np.sin(r_yaw)
        R = np.array([[c, -s], [s, c]])
        def to_glob(p): return (R @ p) + np.array([rx, ry])
        
        P0, P1, P2, P3 = to_glob(P0_loc), to_glob(P1_loc), to_glob(P2_loc), to_glob(P3_loc)
        
        # Échantillonnage complet avec données cinématiques
        t_vals = np.linspace(0, T_opt, num_points)
        pos_list, vel_list, acc_list = [], [], []
        
        for t in t_vals:
            # Position
            u = t / T_opt
            mu = 1 - u
            pt = mu**3*P0 + 3*u*mu**2*P1 + 3*u**2*mu*P2 + u**3*P3
            
            # Cinématique
            v_vec, a_vec = self.get_bezier_derivatives(t, P0, P1, P2, P3, T_opt)
            
            pos_list.append(pt)
            #vel_list.append(np.linalg.norm(v_vec)) # Vitesse scalaire
            #acc_list.append(np.linalg.norm(a_vec)) # Accélération scalaire
            
            
            
            v_norm = np.linalg.norm(v_vec)
            vel_list.append(v_norm)

            # Calcul Accélération Tangentielle (Signée : Positif=Accél, Négatif=Frein)
            # Formule : a_tan = (v . a) / |v|
            if v_norm > 1e-6:
                a_tan = np.dot(v_vec, a_vec) / v_norm
            else:
                a_tan = 0.0 # Indéfini à l'arrêt, on met 0 pour l'affichage
            acc_list.append(a_tan)
            
            
            
            
        return np.array(pos_list), np.array(vel_list), np.array(acc_list), t_vals, (P0,P1,P2,P3), T_opt

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    try:
        ctrl = VelocityController()
        
        # --- SCÉNARIO (Le même que votre image) ---
        # Le robot part vers la droite (+X) mais doit aller en haut (+Y)
        # Il arrive avec une vitesse nulle (parking) ou non
        
        ROBOT = [0.0, 0.0, np.radians(0), 1.5]     # x, y, yaw(0°), v=1.5 m/s
        TARGET = [5.0, 1.0, np.radians(90), 3.4]  # x, y, yaw(90°), v=0.0 m/s
        
        # Calcul
        path, vels, accs, times, ctrls, T_final = ctrl.compute_trajectory(ROBOT, TARGET)
        
        # --- VISUALISATION COMPLÈTE ---
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 2)
        
        # 1. VUE TOP-DOWN (Trajectoire)
        ax_traj = fig.add_subplot(gs[:, 0])
        ax_traj.plot(path[:,0], path[:,1], 'b-', linewidth=2.5, label="Trajectoire Optimisée")
        
        P0, P1, P2, P3 = ctrls
        ax_traj.plot([P0[0], P1[0]], [P0[1], P1[1]], 'k--', alpha=0.3)
        ax_traj.plot([P2[0], P3[0]], [P2[1], P3[1]], 'k--', alpha=0.3)
        
        # Départ / Arrivée
        SCALE_ARROW = 0.5
        
        ax_traj.scatter(*ROBOT[:2], c='g', s=100, zorder=5, label='Start')
        #ax_traj.arrow(ROBOT[0], ROBOT[1], np.cos(ROBOT[2]), np.sin(ROBOT[2]), color='g', width=0.03)
        v_start_len = ROBOT[3] * SCALE_ARROW
        if v_start_len > 0.05: # On ne dessine pas si vitesse nulle
            ax_traj.arrow(ROBOT[0], ROBOT[1], np.cos(ROBOT[2])*v_start_len, np.sin(ROBOT[2])*v_start_len, 
                          color='g', width=0.03, head_width=0.1, length_includes_head=True)
        
        
        ax_traj.scatter(*TARGET[:2], c='r', s=100, zorder=5, label='Target')
        #ax_traj.arrow(TARGET[0], TARGET[1], np.cos(TARGET[2]), np.sin(TARGET[2]), color='r', width=0.03)#
        v_end_len = TARGET[3] * SCALE_ARROW
        if v_end_len > 0.05:
            ax_traj.arrow(TARGET[0], TARGET[1], np.cos(TARGET[2])*v_end_len, np.sin(TARGET[2])*v_end_len, 
                          color='r', width=0.03, head_width=0.1, length_includes_head=True)
        
        
        
        
        ax_traj.axis('equal')
        ax_traj.grid(True)
        ax_traj.set_title(f"Trajectoire XY (Durée: {T_final:.2f}s)")
        ax_traj.legend()
        
        # 2. PROFIL VITESSE
        ax_vel = fig.add_subplot(gs[0, 1])
        ax_vel.plot(times, vels, 'g-', linewidth=2)
        ax_vel.axhline(V_MAX_ROBOT, color='r', linestyle='--', label=f'Limit ({V_MAX_ROBOT} m/s)')
        ax_vel.set_ylabel("Vitesse (m/s)")
        ax_vel.set_title("Profil de Vitesse")
        ax_vel.grid(True)
        ax_vel.legend()
        
        # 3. PROFIL ACCÉLÉRATION
        ax_acc = fig.add_subplot(gs[1, 1])
        #ax_acc.plot(times, accs, 'm-', linewidth=2)
        ax_acc.plot(times, accs, 'm-', linewidth=2, label="Accél. Tangentielle")
        #ax_acc.axhline(A_MAX_ROBOT, color='r', linestyle='--', label=f'Limit ({A_MAX_ROBOT} m/s²)')
        ax_acc.axhline(A_MAX_ROBOT, color='r', linestyle='--', label=f'Lim Acc (+{A_MAX_ROBOT})')
        ax_acc.axhline(-A_MAX_ROBOT, color='orange', linestyle='--', label=f'Lim Frein (-{A_MAX_ROBOT} ?)')
        ax_acc.set_ylabel("Accel (m/s²)")
        ax_acc.set_xlabel("Temps (s)")
        ax_acc.set_title("Profil d'Accélération")
        ax_acc.grid(True)
        ax_acc.legend(fontsize='small')

        # Info text
        plt.figtext(0.55, 0.1, 
                    f"INFO DEBUG:\n"
                    f"Vitesse Départ: {ROBOT[3]} m/s\n"
                    f"Vitesse Arrivée: {TARGET[3]} m/s\n"
                    f"Temps Optimal: {T_final:.3f} s\n"
                    f"Max V mesuré: {np.max(vels):.2f} m/s\n"
                    f"Max A mesuré: {np.max(accs):.2f} m/s²",
                    bbox=dict(facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erreur d'exécution : {e}")