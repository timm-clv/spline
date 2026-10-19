"""
benchmark.py — Vérifie que le pipeline 3D tient 50 Hz (budget 20 ms / pas).
"""
import time
import numpy as np
import torch

from application import VelocityController, TimePredictorNet, TrajectoryExecutor

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def make_controller():
    """Charge le modèle si dispo, sinon weights aléatoires (timing uniquement)."""
    try:
        return VelocityController(), "modèle entraîné 3D chargé"
    except Exception as e:
        print(f"[WARN] {e}")
        print("[WARN] Modèle introuvable -> weights aléatoires (timing seulement)")
        ctrl = object.__new__(VelocityController)
        ctrl.device = torch.device('cpu')
        ctrl.model = TimePredictorNet()
        ctrl.model.eval()
        # 8 entrées : x,y,z, dx,dy,dz, v_start, v_end
        ctrl.X_mean = np.zeros(8, dtype=np.float32)
        ctrl.X_std  = np.ones(8,  dtype=np.float32)
        ctrl.Y_mean = np.array([2.0], dtype=np.float32)
        ctrl.Y_std  = np.array([1.0], dtype=np.float32)
        return ctrl, "weights aléatoires (timing seulement)"

def random_scenario_3d(rng):
    """Génère un état Robot et Cible en 3D."""
    # Robot : (x,y,z), (dir_x,dir_y,dir_z), v_start
    robot = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, rng.uniform(0.0, 5.0)]
    
    # Cible : sphère 3D
    r = rng.uniform(0.5, 10.0)
    t_theta = rng.uniform(-np.pi, np.pi)
    t_phi = rng.uniform(-np.pi/2, np.pi/2)
    tx = r * np.cos(t_phi) * np.cos(t_theta)
    ty = r * np.cos(t_phi) * np.sin(t_theta)
    tz = r * np.sin(t_phi)
    
    # Direction cible 3D aléatoire
    vx, vy, vz = rng.uniform(-1,1), rng.uniform(-1,1), rng.uniform(-1,1)
    norm = np.linalg.norm([vx, vy, vz]) + 1e-6
    target = [tx, ty, tz, vx/norm, vy/norm, vz/norm, rng.uniform(0.0, 5.0)]
    
    return robot, target

def timeit(fn, n_runs, warmup=20):
    """Retourne (p50, p95, p99) en ms."""
    for _ in range(warmup):
        fn()
    times = np.empty(n_runs, dtype=np.float64)
    for i in range(n_runs):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times[i] = (t1 - t0) / 1e6
    return (np.percentile(times, 50),
            np.percentile(times, 95),
            np.percentile(times, 99))

def fmt(name, p50, p95, p99, budget=20.0):
    ok = "✅" if p99 < budget else "❌"
    print(f"  {ok} {name:<38s} p50={p50:6.3f} ms  p95={p95:6.3f} ms  p99={p99:6.3f} ms")

# ---------------------------------------------------------------------------
# Benchmarks unitaires
# ---------------------------------------------------------------------------
def bench_components(ctrl, rng):
    print("\n=== BENCHMARK PAR BRIQUE (3D) ===")
    robot, target = random_scenario_3d(rng)

    # 1. NN forward seul
    def nn_only():
        # On reproduit l'étape de prédiction locale de compute_plan
        robot_pos = np.array(robot[:3])
        target_pos = np.array(target[:3])
        r_dir = np.array(robot[3:6])
        t_dir = np.array(target[3:6])
        R = ctrl.get_local_frame(r_dir)
        delta_loc = R @ (target_pos - robot_pos)
        t_dir_loc = R @ t_dir
        
        inp = np.array([*delta_loc, *t_dir_loc, robot[6], target[6]], dtype=np.float32)
        inp_norm = (inp - ctrl.X_mean) / (ctrl.X_std + 1e-6)
        with torch.no_grad():
            ctrl.model(torch.tensor(inp_norm).unsqueeze(0))

    fmt("NN forward seul (3D)", *timeit(nn_only, 1000))

    # 2. compute_plan complet
    def plan_only():
        ctrl.compute_plan(robot, target)
    fmt("compute_plan (NN + physique 3D)", *timeit(plan_only, 500))

    # 3. sample_plan seul
    plan = ctrl.compute_plan(robot, target)
    def sample_only():
        ctrl.sample_plan(plan, 0.5 * plan['T_opt'])
    fmt("sample_plan (évaluation pure 3D)", *timeit(sample_only, 10000))

    # 4. Pas complet : replan + sample
    def step_replan():
        p = ctrl.compute_plan(robot, target)
        ctrl.sample_plan(p, 0.0)
    fmt("step replan (plan + sample 3D)", *timeit(step_replan, 500))

    # 5. Pas caché : sample d'un plan préexistant
    def step_cached():
        ctrl.sample_plan(plan, 0.5 * plan['T_opt'])
    fmt("step cached (sample seul 3D)", *timeit(step_cached, 10000))

# ---------------------------------------------------------------------------
# Démos 50 Hz
# ---------------------------------------------------------------------------
def demo_50hz_cached(ctrl, rng):
    print("\n=== DÉMO 50 Hz — MODE CACHED (exécution 3D) ===")
    robot, target = random_scenario_3d(rng)
    plan = ctrl.compute_plan(robot, target)
    T_opt = plan['T_opt']
    dt = 0.02
    n_steps = int(np.ceil(T_opt / dt))

    executor = TrajectoryExecutor(ctrl, dt=dt)
    executor.set_target(robot, target)

    latencies = []
    for _ in range(n_steps):
        t0 = time.perf_counter_ns()
        executor.step()
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)

    latencies = np.array(latencies)
    print(f"  Cible        : T_opt = {T_opt:.2f} s -> {n_steps} pas")
    print(f"  Latence      : p50={np.percentile(latencies,50):.3f} ms"
          f" | p99={np.percentile(latencies,99):.3f} ms"
          f" | max={latencies.max():.3f} ms")
    print(f"  Fréquence tenue : {'✅ OK' if latencies.max() < 20 else '❌ FAIL'}")
    return latencies.max() < 20

def demo_50hz_replan(ctrl, rng, n_steps=1000):
    print("\n=== DÉMO 50 Hz — MODE REPLAN (replanification à chaud 3D) ===")
    # Scénario fixe pour la répétabilité
    robot = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0]
    target = [5.0, 3.0, 4.0, 0.0, 0.0, 1.0, 0.5]
    dt = 0.02
    latencies = []

    executor = TrajectoryExecutor(ctrl, dt=dt)
    for _ in range(n_steps):
        t0 = time.perf_counter_ns()
        # On simule un léger mouvement du robot en 3D
        robot[0] += 0.05 * dt
        robot[1] += 0.02 * dt
        robot[2] += 0.03 * dt
        executor.step(robot_state=robot, target_state=target)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)

    latencies = np.array(latencies)
    print(f"  {n_steps} pas de replanification")
    print(f"  Latence      : p50={np.percentile(latencies,50):.3f} ms"
          f" | p95={np.percentile(latencies,95):.3f} ms"
          f" | p99={np.percentile(latencies,99):.3f} ms"
          f" | max={latencies.max():.3f} ms")
    print(f"  Fréquence tenue : {'✅ OK' if latencies.max() < 20 else '❌ FAIL'}")
    return latencies.max() < 20

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print(" BENCHMARK 50 Hz — pipeline Bézier-cubique 3D")
    print(" Budget : 20 ms / pas")
    print("=" * 70)

    ctrl, status = make_controller()
    print(f"[MODE] {status}")

    rng = np.random.default_rng(seed=42)

    bench_components(ctrl, rng)
    ok_cached = demo_50hz_cached(ctrl, rng)
    ok_replan = demo_50hz_replan(ctrl, rng, n_steps=1000)

    print("\n" + "=" * 70)
    if ok_cached and ok_replan:
        print(" ✅ POC 3D VALIDÉ — le pipeline tient 50 Hz dans les deux modes")
    else:
        print(" ❌ POC 3D À OPTIMISER — voir les briques au-dessus du budget")
    print("=" * 70)