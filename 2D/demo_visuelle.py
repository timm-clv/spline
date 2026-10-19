"""
demo_visuelle.py — Animation 50 Hz du POC Bézier-cubique.

Affiche en temps réel :
  - la trajectoire planifiée et le robot qui la parcourt,
  - les profils de vitesse et d'accélération tangentielle,
  - les points de contrôle Bézier.

Usage :
    python demo_visuelle.py          # affichage interactif
    python demo_visuelle.py --save   # sauvegarde aussi demo.gif
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from application import VelocityController, V_MAX_ROBOT, A_MAX_ROBOT


def run_demo(ctrl, robot_state, target_state, save=False, scenario_name=""):
    print(f"[DEMO] Scénario : {scenario_name}")
    print(f"       Robot  : {robot_state}")
    print(f"       Cible  : {target_state}")

    # --- 1. Plan + échantillonnage ---
    plan = ctrl.compute_plan(robot_state, target_state)
    T_opt = plan['T_opt']
    dt = 0.02
    n_frames = int(np.ceil(T_opt / dt)) + 1

    t_vals = np.linspace(0, T_opt, n_frames)
    pos_arr = np.empty((n_frames, 2))
    vel_arr = np.empty(n_frames)
    acc_arr = np.empty(n_frames)
    vx_arr = np.empty(n_frames)
    vy_arr = np.empty(n_frames)
    for i, t in enumerate(t_vals):
        p, v, a = ctrl.sample_plan(plan, t)
        pos_arr[i] = p
        vn = np.linalg.norm(v)
        vel_arr[i] = vn
        acc_arr[i] = np.dot(v, a) / vn if vn > 1e-6 else 0.0
        vx_arr[i] = v[0]
        vy_arr[i] = v[1]

    # --- 2. Figure ---
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1],
                          hspace=0.3, wspace=0.2)
    ax_xy = fig.add_subplot(gs[0, :])
    ax_v  = fig.add_subplot(gs[1, 0])
    ax_a  = fig.add_subplot(gs[1, 1])

    fig.suptitle(f"Démo POC — Trajectoire Bézier cubique contrainte "
                 f"(T_opt = {T_opt:.2f} s)", fontsize=13)

    # --- 3. Vue XY ---
    ax_xy.plot(pos_arr[:, 0], pos_arr[:, 1], 'b--', alpha=0.35,
               linewidth=1.5, label='Plan complet')

    P = [plan['P0'], plan['P1'], plan['P2'], plan['P3']]
    ax_xy.plot([p[0] for p in P], [p[1] for p in P], 'k:', alpha=0.5,
               linewidth=1, label='Polygone de contrôle')
    ax_xy.scatter([p[0] for p in P], [p[1] for p in P],
                  c='orange', s=80, marker='s', zorder=4,
                  edgecolors='black', linewidths=1, label='P0..P3')

    traj_line, = ax_xy.plot([], [], 'b-', linewidth=3, label='Parcouru')
    robot_dot, = ax_xy.plot([], [], 'go', markersize=14,
                            markeredgecolor='darkgreen', markeredgewidth=2,
                            label='Robot', zorder=6)
    ax_xy.plot([robot_state[0]], [robot_state[1]], 'ks',
               markersize=10, alpha=0.6, label='Départ')
    ax_xy.plot([target_state[0]], [target_state[1]], 'rx',
               markersize=18, markeredgewidth=3,
               label='Cible', zorder=6)

    vel_q = ax_xy.quiver([robot_state[0]], [robot_state[1]],
                         [0.0], [0.0],
                         angles='xy', scale_units='xy', scale=1.0,
                         color='green', alpha=0.8, width=0.006,
                         headwidth=5, headlength=6, zorder=7)

    margin = 1.2
    ax_xy.set_xlim(min(pos_arr[:, 0].min(), robot_state[0],
                       target_state[0]) - margin,
                   max(pos_arr[:, 0].max(), robot_state[0],
                       target_state[0]) + margin)
    ax_xy.set_ylim(min(pos_arr[:, 1].min(), robot_state[1],
                       target_state[1]) - margin,
                   max(pos_arr[:, 1].max(), robot_state[1],
                       target_state[1]) + margin)
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Trajectoire planifiée (vue du dessus)', fontsize=11)
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis('equal')
    ax_xy.legend(loc='best', fontsize=9)

    # --- 4. V(t) ---
    ax_v.plot(t_vals, vel_arr, 'g-', alpha=0.3, linewidth=1)
    vel_line, = ax_v.plot([], [], 'g-', linewidth=2.5)
    vel_cursor, = ax_v.plot([], [], 'ro', markersize=8, zorder=5)
    ax_v.axhline(V_MAX_ROBOT, color='r', linestyle='--', alpha=0.7,
                 label=f'V_max = {V_MAX_ROBOT} m/s')
    ax_v.set_xlim(0, T_opt)
    ax_v.set_ylim(0, max(V_MAX_ROBOT * 1.1, vel_arr.max() * 1.15))
    ax_v.set_xlabel('Temps (s)')
    ax_v.set_ylabel('Vitesse (m/s)')
    ax_v.set_title('Profil de vitesse', fontsize=11)
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc='best', fontsize=9)

    # --- 5. a(t) ---
    ax_a.plot(t_vals, acc_arr, 'm-', alpha=0.3, linewidth=1)
    acc_line, = ax_a.plot([], [], 'm-', linewidth=2.5)
    acc_cursor, = ax_a.plot([], [], 'ro', markersize=8, zorder=5)
    ax_a.axhline(A_MAX_ROBOT, color='r', linestyle='--', alpha=0.7,
                 label=f'A_max = {A_MAX_ROBOT} m/s²')
    ax_a.axhline(-A_MAX_ROBOT, color='orange', linestyle='--', alpha=0.7)
    ax_a.axhline(0, color='gray', alpha=0.5, linewidth=0.5)
    ax_a.set_xlim(0, T_opt)
    y_lim = max(A_MAX_ROBOT * 1.15, np.abs(acc_arr).max() * 1.15)
    ax_a.set_ylim(-y_lim, y_lim)
    ax_a.set_xlabel('Temps (s)')
    ax_a.set_ylabel('Accél. tangentielle (m/s²)')
    ax_a.set_title("Profil d'accélération tangentielle", fontsize=11)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc='best', fontsize=9)

    info = fig.text(0.5, 0.94, '', ha='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='lightyellow',
                              edgecolor='gray', alpha=0.9))

    vel_scale = 0.25  # longueur de la flèche vitesse

    # --- 6. Boucle d'animation ---
    def update(frame):
        if frame >= n_frames:
            frame = n_frames - 1
        t = t_vals[frame]
        p = pos_arr[frame]

        traj_line.set_data(pos_arr[:frame+1, 0], pos_arr[:frame+1, 1])
        robot_dot.set_data([p[0]], [p[1]])

        vel_line.set_data(t_vals[:frame+1], vel_arr[:frame+1])
        vel_cursor.set_data([t], [vel_arr[frame]])
        acc_line.set_data(t_vals[:frame+1], acc_arr[:frame+1])
        acc_cursor.set_data([t], [acc_arr[frame]])

        # Flèche vitesse
        vel_q.set_offsets(np.array([[p[0], p[1]]]))
        if vel_arr[frame] > 0.05:
            vel_q.set_UVC(np.array([vx_arr[frame] * vel_scale]),
                          np.array([vy_arr[frame] * vel_scale]))
            vel_q.set_visible(True)
        else:
            vel_q.set_visible(False)

        info.set_text(
            f"t = {t:.2f} / {T_opt:.2f} s   |   "
            f"V = {vel_arr[frame]:.2f} m/s   |   "
            f"a_tan = {acc_arr[frame]:+.2f} m/s²   |   "
            f"frame {frame+1}/{n_frames}"
        )
        return (traj_line, robot_dot, vel_line, vel_cursor,
                acc_line, acc_cursor, vel_q, info)

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=20, repeat=False, blit=False)

    if save:
        out = f'demo_{scenario_name or "default"}.gif'
        try:
            print(f"[SAVE] Écriture de {out} (peut prendre ~10 s)...")
            anim.save(out, writer=PillowWriter(fps=50))
            print(f"[SAVE] {out} sauvegardé.")
        except Exception as e:
            print(f"[WARN] Sauvegarde impossible : {e}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return anim


if __name__ == "__main__":
    save = '--save' in sys.argv
    ctrl = VelocityController()

    run_demo(
        ctrl,
        robot_state=[0.0, 0.0, 0.0, 2.0],       # x, y, yaw, v_start
        target_state=[-2.0, -1.0, np.pi/2, 0.5],  # x, y, yaw, v_end
        save=save,
        scenario_name="statique",
    )