"""
benchmark.py — Vérifie que le pipeline tient 50 Hz (budget 20 ms / pas).

Usage :
    python benchmark.py

Produit :
    1. Tableau de latence par brique (p50 / p95 / p99)
    2. Deux démos 50 Hz (exécution cachée et replanification à chaud)
    3. Assertion finale : p99 < 20 ms
"""
import time
import numpy as np

from application import VelocityController, TimePredictorNet, TrajectoryExecutor


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def make_controller():
    """Charge le modèle si dispo, sinon weights aléatoires (timing uniquement)."""
    try:
        return VelocityController(), "modèle entraîné chargé"
    except Exception as e:
        print(f"[WARN] {e}")
        print("[WARN] Modèle introuvable → weights aléatoires (timing seulement)")
        import torch
        ctrl = object.__new__(VelocityController)
        ctrl.device = torch.device('cpu')
        ctrl.model = TimePredictorNet()
        ctrl.model.eval()
        ctrl.X_mean = np.zeros(6, dtype=np.float32)
        ctrl.X_std  = np.ones(6,  dtype=np.float32)
        ctrl.Y_mean = np.array([2.0], dtype=np.float32)
        ctrl.Y_std  = np.array([1.0], dtype=np.float32)
        return ctrl, "weights aléatoires (timing seulement)"


def random_scenario(rng):
    robot = [0.0, 0.0, 0.0, rng.uniform(0.0, 3.0)]
    r = rng.uniform(0.5, 10.0)
    a = rng.uniform(-np.pi/2, np.pi/2)
    target = [r*np.cos(a), r*np.sin(a),
              rng.uniform(-np.pi, np.pi), rng.uniform(0.0, 3.0)]
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
    print("\n=== BENCHMARK PAR BRIQUE ===")

    robot, target = random_scenario(rng)

    # 1. NN forward seul (via compute_plan mais on isole en mesurant plusieurs)
    #    On crée un wrapper minimal
    def nn_only():
        rx, ry, r_yaw, v_start = robot
        tx, ty, t_yaw, v_end = target
        x_loc, y_loc, th_loc = ctrl.global_to_local(
            [rx, ry], r_yaw, [tx, ty], t_yaw)
        inp = np.array([x_loc, y_loc, np.cos(th_loc), np.sin(th_loc),
                        v_start, v_end], dtype=np.float32)
        inp_norm = (inp - ctrl.X_mean) / (ctrl.X_std + 1e-6)
        import torch
        with torch.no_grad():
            ctrl.model(torch.tensor(inp_norm).unsqueeze(0))

    fmt("NN forward seul",
        *timeit(nn_only, 1000))

    # 2. compute_plan complet (NN + physique + reconstruction)
    def plan_only():
        ctrl.compute_plan(robot, target)
    fmt("compute_plan (NN + physique)",
        *timeit(plan_only, 500))

    # 3. sample_plan seul
    plan = ctrl.compute_plan(robot, target)
    def sample_only():
        ctrl.sample_plan(plan, 0.5 * plan['T_opt'])
    fmt("sample_plan (évaluation pure)",
        *timeit(sample_only, 10000))

    # 4. Pas complet : replan + sample
    def step_replan():
        p = ctrl.compute_plan(robot, target)
        ctrl.sample_plan(p, 0.0)
    fmt("step replan (plan + sample)",
        *timeit(step_replan, 500))

    # 5. Pas caché : sample d'un plan préexistant
    def step_cached():
        ctrl.sample_plan(plan, 0.5 * plan['T_opt'])
    fmt("step cached (sample seul)",
        *timeit(step_cached, 10000))


# ---------------------------------------------------------------------------
# Démos 50 Hz
# ---------------------------------------------------------------------------
def demo_50hz_cached(ctrl, rng):
    print("\n=== DÉMO 50 Hz — MODE CACHED (exécution) ===")
    robot, target = random_scenario(rng)
    # S'assurer que T_opt n'est pas dégénéré
    plan = ctrl.compute_plan(robot, target)
    T_opt = plan['T_opt']
    dt = 0.02
    n_steps = int(np.ceil(T_opt / dt))

    executor = TrajectoryExecutor(ctrl, dt=dt)
    executor.set_target(robot, target)

    latencies = []
    traj = []
    for _ in range(n_steps):
        t0 = time.perf_counter_ns()
        pos, vel, acc, t = executor.step()
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)
        traj.append(pos)

    latencies = np.array(latencies)
    print(f"  Cible        : T_opt = {T_opt:.2f} s → {n_steps} pas")
    print(f"  Latence      : p50={np.percentile(latencies,50):.3f} ms"
          f" | p99={np.percentile(latencies,99):.3f} ms"
          f" | max={latencies.max():.3f} ms")
    print(f"  Fréquence tenue : {'✅ OK' if latencies.max() < 20 else '❌ FAIL'}")
    return latencies.max() < 20


def demo_50hz_replan(ctrl, rng, n_steps=1000):
    print("\n=== DÉMO 50 Hz — MODE REPLAN (replanification à chaud) ===")
    robot = [0.0, 0.0, 0.0, 2.0]
    target = [5.0, 3.0, np.pi/2, 0.5]
    dt = 0.02
    latencies = []

    executor = TrajectoryExecutor(ctrl, dt=dt)
    for _ in range(n_steps):
        t0 = time.perf_counter_ns()
        # On fait croire que le robot a avancé
        robot[0] += 0.02 * dt
        robot[1] += 0.01 * dt
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
    print(" BENCHMARK 50 Hz — pipeline Bézier-cubique contraint")
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
        print(" ✅ POC VALIDÉ — le pipeline tient 50 Hz dans les deux modes")
    else:
        print(" ❌ POC À OPTIMISER — voir les briques au-dessus du budget")
    print("=" * 70)