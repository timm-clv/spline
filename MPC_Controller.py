import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
  

class AsymmetricVelocityProfiler:
    def __init__(self, v_max, a_max, a_min):
        """
        a_max : accélération max (> 0)
        a_min : décélération max (freinage) (> 0, on gère le signe en interne)
        """
        self.v_max = v_max
        self.a_max = a_max
        self.a_min = a_min # Freinage

    def compute_next_velocity(self, dist_remaining, v_curr, dt):
        """
        Calcule la vitesse consigne pour le prochain pas de temps (MPC).
        Logique : "Bang-Bang" control adouci.
        """
        # 1. Distance nécessaire pour s'arrêter depuis v_curr (d = v^2 / 2a)
        stopping_dist = (v_curr**2) / (2 * self.a_min)
        
        # 2. Décision
        if dist_remaining <= stopping_dist:
            # ZONE DE FREINAGE CRITIQUE : On doit piler
            v_target = max(0, v_curr - self.a_min * dt)
        elif v_curr < self.v_max:
            # ZONE D'ACCÉLÉRATION : On a de la marge
            v_target = min(self.v_max, v_curr + self.a_max * dt)
        else:
            # ZONE DE CROISIÈRE : On maintient ou on ralentit doucement vers v_max
            v_target = max(self.v_max, v_curr - self.a_min * dt)
            
        return v_target

    def generate_full_profile(self, total_length, v_start, v_end=0, dt=0.05):
        """
        Génère s(t) complet pour interpolation (Optionnel pour visualisation)
        C'est complexe mathématiquement (cas triangulaire vs trapèze asymétrique).
        Pour le MPC, on utilise surtout compute_next_velocity.
        """
        pass # Non critique pour le MPC temps réel "step-by-step"
    
    


# ==============================================================================
# 1. MODÈLE (Toujours identique)
# ==============================================================================
class BezierNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x): return self.net(x)

# ==============================================================================
# 2. CONTROLEUR MPC
# ==============================================================================
class MPCController:
    def __init__(self, model_path, stats_path, v_max=1.0, a_max=1.0):
        # Chargement NN
        self.model = BezierNet()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        self.model.eval()
        
        # Chargement Stats
        stats = np.load(stats_path)
        self.X_mean, self.X_std = stats['X_mean'], stats['X_std']
        self.Y_mean, self.Y_std = stats['Y_mean'], stats['Y_std']
        
        # Physique
        self.profiler = AsymmetricVelocityProfiler(v_max=v_max, a_max=a_max, a_min=a_max)

    def get_control_command(self, robot_state, target_state, dt=0.1):
        """
        BOUCLE PRINCIPALE DU MPC
        robot_state: [x, y, yaw, v_current]
        target_state: [x, y, yaw]
        """
        rx, ry, ryaw, v_curr = robot_state
        tx, ty, tyaw = target_state
        
        # --- A. TRANSFORMATION LOCALE ---
        dx, dy = tx - rx, ty - ry
        c, s = np.cos(ryaw), np.sin(ryaw)
        x_loc = c * dx + s * dy
        y_loc = -s * dx + c * dy
        th_loc = (tyaw - ryaw + np.pi) % (2*np.pi) - np.pi
        
        # --- B. INFÉRENCE NN (Géométrie) ---
        inp = np.array([x_loc, y_loc, np.cos(th_loc), np.sin(th_loc)], dtype=np.float32)
        inp_norm = (inp - self.X_mean) / (self.X_std + 1e-6)
        with torch.no_grad():
            out = self.model(torch.tensor(inp_norm))
        k1_nn, k2_nn, T_nn = out.numpy() * (self.Y_std + 1e-6) + self.Y_mean
        
        # --- C. CORRECTION MPC (Le cœur du problème) ---
        # Le NN a appris k1 pour un départ arrêté. 
        # Si on bouge, la physique impose : P1 = P0 + (v * T_nn / 3) * u_tangente
        # En local, u_tangente est toujours (1, 0).
        
        if v_curr > 0.1:
            # On force la continuité de la vitesse
            # k1 représente la "puissance" du vecteur tangent initial
            k1_physique = (v_curr * T_nn) / 3.0
            
            # Mélange : On fait confiance à la physique, mais on garde des bornes
            # Si T_nn est mal prédit, k1 peut être faux. 
            k1_final = k1_physique
        else:
            k1_final = max(0.1, k1_nn) # Départ arrêté, on fait confiance au NN

        k2_final = max(0.1, k2_nn)
        
        # --- D. RECONSTRUCTION DE LA COURBE LOCALE ---
        P0 = np.array([0., 0.])
        P1 = np.array([k1_final, 0.]) # Tangent à l'axe X local
        P3 = np.array([x_loc, y_loc])
        dir_end = np.array([np.cos(th_loc), np.sin(th_loc)])
        P2 = P3 - k2_final * dir_end
        
        # Estimation longueur d'arc (rapide)
        chord = np.linalg.norm(P3 - P0)
        arc_length = chord * 1.1 # Approximation ou calcul précis par échantillonnage
        
        # --- E. PROFIL DE VITESSE (Cinématique) ---
        # Quelle vitesse doit-on avoir au prochain pas ?
        v_next_cmd = self.profiler.compute_next_velocity(arc_length, v_curr, dt)
        
        # --- F. CALCUL DE LA POSITION SUIVANTE (Consigne) ---
        # On avance sur la courbe de Bézier d'une distance d = v_avg * dt
        dist_step = ((v_curr + v_next_cmd) / 2) * dt
        
        # On cherche u tel que dist(0->u) = dist_step
        # Approximation linéaire pour petit dt : u approx = dist_step / arc_length
        u_next = dist_step / (arc_length + 1e-6)
        u_next = min(1.0, u_next)
        
        # Calcul point local Bezier
        mu = 1 - u_next
        pos_loc_next = (mu**3 * P0 + 3 * u_next * mu**2 * P1 + 3 * u_next**2 * mu * P2 + u_next**3 * P3)
        
        # --- G. RETOUR AU GLOBAL ---
        # On renvoie la consigne de vitesse et le point cible global
        pos_glob_next = np.array([
            c * pos_loc_next[0] - s * pos_loc_next[1] + rx,
            s * pos_loc_next[0] + c * pos_loc_next[1] + ry
        ])
        
        return v_next_cmd, pos_glob_next, (P0, P1, P2, P3) # Debug info

# ==============================================================================
# 3. SIMULATION
# ==============================================================================
if __name__ == "__main__":
    try:
        mpc = MPCController("bezier_model.pth", "stats_light.npz", v_max=1.5, a_max=0.8)
        
        # État initial
        robot_state = np.array([0.0, 0.0, 1.0, 1.0]) # x, y, yaw, v
        target_state = np.array([5.0, 2.0, np.radians(90)])
        
        DT = 0.1
        history_x, history_y, history_v = [], [], []
        
        print("Début Simulation MPC...")
        for step in range(100): # 10 secondes max
            
            # 1. Calcul MPC
            v_cmd, pos_next_glob, debug_pts = mpc.get_control_command(robot_state, target_state, DT)
            
            # 2. Mise à jour Robot (Simulation physique simple)
            # Le robot va vers le point calculé
            dx = pos_next_glob[0] - robot_state[0]
            dy = pos_next_glob[1] - robot_state[1]
            wanted_yaw = np.arctan2(dy, dx)
            
            # Mise à jour position
            robot_state[0] = pos_next_glob[0]
            robot_state[1] = pos_next_glob[1]
            # Mise à jour vitesse (le robot obéit parfaitement à la consigne ici)
            robot_state[3] = v_cmd
            # Mise à jour angle (on suppose qu'il tourne instantanément pour l'exemple)
            robot_state[2] = wanted_yaw 
            
            # Logs
            history_x.append(robot_state[0])
            history_y.append(robot_state[1])
            history_v.append(robot_state[3])
            
            dist_to_goal = np.linalg.norm(target_state[:2] - robot_state[:2])
            if dist_to_goal < 0.1 and v_cmd < 0.1:
                print(f"Cible atteinte en {step*DT:.1f}s")
                break
        
        # --- VISUALISATION ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Trajectoire XY
        ax1.plot(history_x, history_y, 'b.-', label="Trajectoire MPC")
        ax1.scatter(0,0, c='g', label='Départ')
        ax1.scatter(target_state[0], target_state[1], c='r', marker='x', label='Cible')
        ax1.arrow(target_state[0], target_state[1], np.cos(target_state[2]), np.sin(target_state[2]), color='r', head_width=0.2)
        ax1.grid(True)
        ax1.axis('equal')
        ax1.set_title("Trajectoire Robot")
        ax1.legend()
        
        # Profil Vitesse
        ax2.plot(np.arange(len(history_v))*DT, history_v, 'k-')
        ax2.set_title("Profil de Vitesse Généré")
        ax2.set_xlabel("Temps (s)")
        ax2.set_ylabel("Vitesse (m/s)")
        ax2.grid(True)
        
        plt.show()
        
    except Exception as e:
        print(f"Erreur: {e}")
        print("Vérifiez que bezier_model.pth et stats_light.npz sont présents.")
