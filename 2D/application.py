import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# ==============================================================================
# CONFIGURATION ROBOT
# ==============================================================================
V_MAX_ROBOT = 6.5   # m/s
A_MAX_ROBOT = 6.0   # m/s²  

# ==============================================================================
# 1. ARCHITECTURE DU MODÈLE
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

# ==============================================================================
# 2. CONTRÔLEUR INTELLIGENT
# ==============================================================================
class VelocityController:
    def __init__(self, model_path="modele_temps.pth",
                 stats_path="statistique_de_normalisation.npz"):
        self.device = torch.device('cpu')

        self.model = TimePredictorNet().to(self.device)
        try:
            state_dict = torch.load(model_path, map_location=self.device,
                                    weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"[INIT] Modèle chargé : {model_path}")
        except Exception as e:
            print(f"[ERREUR] Impossible de charger le modèle : {e}")
            raise

        try:
            stats = np.load(stats_path)
            self.X_mean, self.X_std = stats['X_mean'], stats['X_std']
            self.Y_mean, self.Y_std = stats['Y_mean'], stats['Y_std']
            print(f"[INIT] Stats chargées.")
        except FileNotFoundError:
            raise FileNotFoundError("statistique_de_normalisation.npz manquant.")

    # -------------------------------------------------------------------------
    # 2.a  Helpers existants (inchangés)
    # -------------------------------------------------------------------------
    def global_to_local(self, robot_pos, robot_yaw, target_pos, target_yaw):
        dx, dy = target_pos[0] - robot_pos[0], target_pos[1] - robot_pos[1]
        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        x_loc = c * dx + s * dy
        y_loc = -s * dx + c * dy
        theta_loc = (target_yaw - robot_yaw + np.pi) % (2 * np.pi) - np.pi
        return x_loc, y_loc, theta_loc

    def get_bezier_derivatives(self, t, P0, P1, P2, P3, T_total):
        u = t / T_total
        mu = 1 - u
        dP_du = 3*mu**2*(P1-P0) + 6*mu*u*(P2-P1) + 3*u**2*(P3-P2)
        d2P_du2 = 6*mu*(P2-2*P1+P0) + 6*u*(P3-2*P2+P1)
        return dP_du / T_total, d2P_du2 / T_total**2

    def refine_time_with_physics(self, P0, P3, u_start, u_end,
                                 v_start, v_end, T_init,
                                 margin=0.99, n_samples=30,
                                 tol=1e-3, max_iter=40):
        """
        Recherche binaire du T minimal faisable.
        margin=0.99 : 1 % de sécurité (physique robot).
        """
        def feasible(T):
            if T < 0.05:
                return False
            k1 = (v_start * T) / 3.0
            k2 = (v_end   * T) / 3.0
            P1 = P0 + k1 * u_start
            P2 = P3 - k2 * u_end
            for t in np.linspace(0, T, n_samples):
                vel, acc = self.get_bezier_derivatives(t, P0, P1, P2, P3, T)
                if np.linalg.norm(vel) > V_MAX_ROBOT * margin:
                    return False
                if np.linalg.norm(acc) > A_MAX_ROBOT * margin:
                    return False
            return True

        T_high = max(T_init, 0.5)
        for _ in range(20):
            if feasible(T_high): break
            T_high *= 1.5
        T_low = T_high * 0.5
        for _ in range(20):
            if not feasible(T_low): break
            T_low *= 0.5
        for _ in range(max_iter):
            T_mid = 0.5 * (T_low + T_high)
            if feasible(T_mid): T_high = T_mid
            else:               T_low  = T_mid
            if T_high - T_low < tol: break
        return T_high

    # -------------------------------------------------------------------------
    # 2.b  NOUVEAU — planification pure (NN + physique + contrôle points)
    # -------------------------------------------------------------------------
    def compute_plan(self, robot_state, target_state):
        """
        Calcule et retourne un plan complet SANS échantillonnage temporel.
        Utilisé pour la replanification à chaud (une fois par nouvelle cible).
        """
        rx, ry, r_yaw, v_start = robot_state
        tx, ty, t_yaw, v_end = target_state

        # 1. Prédiction NN
        x_loc, y_loc, th_loc = self.global_to_local([rx, ry], r_yaw, [tx, ty], t_yaw)
        inp = np.array([x_loc, y_loc, np.cos(th_loc), np.sin(th_loc),
                        v_start, v_end], dtype=np.float32)
        inp_norm = (inp - self.X_mean) / (self.X_std + 1e-6)
        with torch.no_grad():
            T_pred_norm = self.model(
                torch.tensor(inp_norm).unsqueeze(0).to(self.device)
            ).item()
        T_init = max(0.5, T_pred_norm * self.Y_std[0] + self.Y_mean[0])

        # 2. Contrôle points en repère local
        P0_loc = np.array([0., 0.])
        P3_loc = np.array([x_loc, y_loc])
        u_start_loc = np.array([1., 0.])
        u_end_loc = np.array([np.cos(th_loc), np.sin(th_loc)])

        # 3. Raffinement physique
        T_opt = self.refine_time_with_physics(
            P0_loc, P3_loc, u_start_loc, u_end_loc,
            v_start, v_end, T_init
        )

        # 4. Reconstruction finale + passage global
        k1 = (v_start * T_opt) / 3.0
        k2 = (v_end * T_opt) / 3.0
        P1_loc = P0_loc + k1 * u_start_loc
        P2_loc = P3_loc - k2 * u_end_loc

        c, s = np.cos(r_yaw), np.sin(r_yaw)
        R = np.array([[c, -s], [s, c]])
        offset = np.array([rx, ry])
        to_glob = lambda p: R @ p + offset
        P0, P1, P2, P3 = map(to_glob, (P0_loc, P1_loc, P2_loc, P3_loc))

        return {
            'P0': P0, 'P1': P1, 'P2': P2, 'P3': P3,
            'T_opt': float(T_opt),
            'robot_state': robot_state,
            'target_state': target_state,
        }

    # -------------------------------------------------------------------------
    # 2.c  NOUVEAU — échantillonnage pur (microsecondes)
    # -------------------------------------------------------------------------
    def sample_plan(self, plan, t):
        """
        Retourne (position, vitesse, accélération) du plan à l'instant t.
        Pas d'appel NN, pas de physique — juste de l'évaluation polynomiale.
        """
        P0, P1, P2, P3 = plan['P0'], plan['P1'], plan['P2'], plan['P3']
        T_opt = plan['T_opt']
        t = float(np.clip(t, 0.0, T_opt))
        u = t / T_opt
        mu = 1 - u
        pos = mu**3*P0 + 3*u*mu**2*P1 + 3*u**2*mu*P2 + u**3*P3
        vel, acc = self.get_bezier_derivatives(t, P0, P1, P2, P3, T_opt)
        return pos, vel, acc

    # -------------------------------------------------------------------------
    # 2.d  Rétro-compat : compute_trajectory = plan + N samples
    # -------------------------------------------------------------------------
    def compute_trajectory(self, robot_state, target_state, num_points=100):
        plan = self.compute_plan(robot_state, target_state)
        T_opt = plan['T_opt']

        t_vals = np.linspace(0, T_opt, num_points)
        pos_list, vel_list, acc_list = [], [], []
        for t in t_vals:
            p, v, a = self.sample_plan(plan, t)
            v_norm = float(np.linalg.norm(v))
            vel_list.append(v_norm)
            acc_list.append(float(np.dot(v, a) / v_norm) if v_norm > 1e-6 else 0.0)
            pos_list.append(p)

        ctrls = (plan['P0'], plan['P1'], plan['P2'], plan['P3'])
        return np.array(pos_list), np.array(vel_list), np.array(acc_list), \
               t_vals, ctrls, T_opt

# ==============================================================================
# 3. NOUVEAU — Exécuteur 50 Hz (boucle de contrôle)
# ==============================================================================
class TrajectoryExecutor:
    """
    Exécute un plan à fréquence fixe (par défaut 50 Hz = 20 ms).
    Deux modes :
      - mode 'cached' : planifier une fois, échantillonner à chaque pas
      - mode 'replan' : replanifier à chaque pas (test de robustesse)
    """
    def __init__(self, controller, dt=0.02):
        self.ctrl = controller
        self.dt = dt
        self.plan = None
        self.t = 0.0

    def set_target(self, robot_state, target_state):
        self.plan = self.ctrl.compute_plan(robot_state, target_state)
        self.t = 0.0

    def is_done(self):
        return self.plan is not None and self.t >= self.plan['T_opt']

    def step(self, robot_state=None, target_state=None):
        """
        Avance d'un pas dt.
        Si robot_state/target_state fournis → mode replan.
        Sinon → mode cached (utilise le plan courant).
        Retourne (pos, vel, acc, t).
        """
        if robot_state is not None and target_state is not None:
            self.plan = self.ctrl.compute_plan(robot_state, target_state)
            self.t = 0.0

        if self.plan is None:
            raise RuntimeError("Aucun plan actif. Appelez set_target d'abord.")

        self.t = min(self.t + self.dt, self.plan['T_opt'])
        pos, vel, acc = self.ctrl.sample_plan(self.plan, self.t)
        return pos, vel, acc, self.t