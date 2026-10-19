import numpy as np
import torch
import torch.nn as nn

V_MAX_ROBOT = 6.5
A_MAX_ROBOT = 6.0

class TimePredictorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 20), nn.ReLU(),
            nn.Linear(20, 10), nn.ReLU(),
            nn.Linear(10, 1),
        )
    def forward(self, x):
        return self.net(x)

class VelocityController:
    def __init__(self, model_path="modele_temps.pth", stats_path="statistique_de_normalisation.npz"):
        self.device = torch.device('cpu')
        self.model = TimePredictorNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        stats = np.load(stats_path)
        self.X_mean, self.X_std = stats['X_mean'], stats['X_std']
        self.Y_mean, self.Y_std = stats['Y_mean'], stats['Y_std']

    def get_local_frame(self, v_dir):
        """Génère une base orthonormée locale à partir du vecteur directeur."""
        u = v_dir / (np.linalg.norm(v_dir) + 1e-8)
        # Éviter les cas dégénérés (si le robot pointe vers le Z global)
        if abs(u[2]) < 1 - 1e-5:
            v = np.cross([0, 0, 1], u)
        else:
            v = np.cross([0, 1, 0], u)
        v /= np.linalg.norm(v)
        w = np.cross(u, v)
        return np.array([u, v, w]) # Matrice R (3x3)

    def get_bezier_derivatives(self, t, P0, P1, P2, P3, T_total):
        u = t / T_total
        mu = 1 - u
        dP_du = 3*mu**2*(P1-P0) + 6*mu*u*(P2-P1) + 3*u**2*(P3-P2)
        d2P_du2 = 6*mu*(P2-2*P1+P0) + 6*u*(P3-2*P2+P1)
        return dP_du / T_total, d2P_du2 / T_total**2

    def refine_time_with_physics(self, P0, P3, u_start, u_end, v_start, v_end, T_init,
                                 margin=0.99, n_samples=30, tol=1e-3, max_iter=40):
        def feasible(T):
            if T < 0.05: return False
            P1 = P0 + (v_start * T / 3.0) * u_start
            P2 = P3 - (v_end * T / 3.0) * u_end
            for t in np.linspace(0, T, n_samples):
                vel, acc = self.get_bezier_derivatives(t, P0, P1, P2, P3, T)
                if np.linalg.norm(vel) > V_MAX_ROBOT * margin: return False
                if np.linalg.norm(acc) > A_MAX_ROBOT * margin: return False
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
            else: T_low = T_mid
            if T_high - T_low < tol: break
        return T_high

    def compute_plan(self, robot_state, target_state):
        # robot_state : [x, y, z, v_dir_x, v_dir_y, v_dir_z, v_start]
        rx, ry, rz, rdx, rdy, rdz, v_start = robot_state
        # target_state : [x, y, z, ux, uy, uz, v_end]
        tx, ty, tz, tux, tuy, tuz, v_end = target_state

        robot_pos = np.array([rx, ry, rz])
        target_pos = np.array([tx, ty, tz])
        
        # Gérer le cas où le robot est à l'arrêt (vitesse nulle)
        if v_start > 0.1:
            robot_dir = np.array([rdx, rdy, rdz])
        else:
            robot_dir = np.array([1.0, 0.0, 0.0]) # Direction par défaut
            
        target_dir = np.array([tux, tuy, tuz])

        # 1. Passage en repère local
        R = self.get_local_frame(robot_dir)
        delta_pos_glob = target_pos - robot_pos
        delta_pos_loc = R @ delta_pos_glob
        target_dir_loc = R @ target_dir

        # 2. Prédiction NN
        inp = np.array([*delta_pos_loc, *target_dir_loc, v_start, v_end], dtype=np.float32)
        inp_norm = (inp - self.X_mean) / (self.X_std + 1e-6)
        with torch.no_grad():
            T_pred_norm = self.model(torch.tensor(inp_norm).unsqueeze(0)).item()
        T_init = max(0.5, T_pred_norm * self.Y_std[0] + self.Y_mean[0])

        # 3. Raffinement physique
        P0_loc = np.array([0., 0., 0.])
        u_start = np.array([1., 0., 0.])
        
        T_opt = self.refine_time_with_physics(P0_loc, delta_pos_loc, u_start, target_dir_loc, v_start, v_end, T_init)

        # 4. Reconstruction finale
        P1_loc = P0_loc + (v_start * T_opt / 3.0) * u_start
        P2_loc = delta_pos_loc - (v_end * T_opt / 3.0) * target_dir_loc

        to_glob = lambda p: R.T @ p + robot_pos
        P0, P1, P2, P3 = map(to_glob, (P0_loc, P1_loc, P2_loc, delta_pos_loc))

        return {'P0': P0, 'P1': P1, 'P2': P2, 'P3': P3, 'T_opt': float(T_opt)}

    def sample_plan(self, plan, t):
        P0, P1, P2, P3 = plan['P0'], plan['P1'], plan['P2'], plan['P3']
        T_opt = plan['T_opt']
        t = float(np.clip(t, 0.0, T_opt))
        u = t / T_opt
        mu = 1 - u
        pos = mu**3*P0 + 3*u*mu**2*P1 + 3*u**2*mu*P2 + u**3*P3
        vel, acc = self.get_bezier_derivatives(t, P0, P1, P2, P3, T_opt)
        return pos, vel, acc
    
    
class TrajectoryExecutor:
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
        if robot_state is not None and target_state is not None:
            self.plan = self.ctrl.compute_plan(robot_state, target_state)
            self.t = 0.0
        if self.plan is None:
            raise RuntimeError("Aucun plan actif. Appelez set_target d'abord.")
        self.t = min(self.t + self.dt, self.plan['T_opt'])
        pos, vel, acc = self.ctrl.sample_plan(self.plan, self.t)
        return pos, vel, acc, self.t