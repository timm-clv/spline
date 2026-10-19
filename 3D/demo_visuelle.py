import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from application import VelocityController, V_MAX_ROBOT, A_MAX_ROBOT

def run_demo(ctrl, robot_state, target_state):
    plan = ctrl.compute_plan(robot_state, target_state)
    T_opt = plan['T_opt']
    dt = 0.02
    n_frames = int(np.ceil(T_opt / dt)) + 1

    t_vals = np.linspace(0, T_opt, n_frames)
    pos_arr = np.empty((n_frames, 3))
    vel_arr, acc_arr = np.empty(n_frames), np.empty(n_frames)

    for i, t in enumerate(t_vals):
        p, v, a = ctrl.sample_plan(plan, t)
        pos_arr[i] = p
        vn = np.linalg.norm(v)
        vel_arr[i] = vn
        acc_arr[i] = np.dot(v, a) / vn if vn > 1e-6 else 0.0

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax_xyz = fig.add_subplot(gs[0, :], projection='3d')
    ax_v = fig.add_subplot(gs[1, 0])
    ax_a = fig.add_subplot(gs[1, 1])

    # 3D Plot
    ax_xyz.plot(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], 'b--', alpha=0.3)
    P = [plan['P0'], plan['P1'], plan['P2'], plan['P3']]
    ax_xyz.plot([p[0] for p in P], [p[1] for p in P], [p[2] for p in P], 'k:', alpha=0.5)
    ax_xyz.scatter([p[0] for p in P], [p[1] for p in P], [p[2] for p in P], c='orange', marker='s')
    
    ax_xyz.scatter(*robot_state[:3], c='g', s=100, label='Start')
    ax_xyz.scatter(*target_state[:3], c='r', marker='x', s=100, label='Target')

    traj_line, = ax_xyz.plot([], [], [], 'b-', linewidth=3)
    robot_dot, = ax_xyz.plot([], [], [], 'go', markersize=10)

    # Limites 3D
    max_range = np.array([pos_arr[:,0].max()-pos_arr[:,0].min(), 
                          pos_arr[:,1].max()-pos_arr[:,1].min(), 
                          pos_arr[:,2].max()-pos_arr[:,2].min()]).max() / 2.0
    mid_x = (pos_arr[:,0].max()+pos_arr[:,0].min()) * 0.5
    mid_y = (pos_arr[:,1].max()+pos_arr[:,1].min()) * 0.5
    mid_z = (pos_arr[:,2].max()+pos_arr[:,2].min()) * 0.5
    ax_xyz.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_xyz.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_xyz.set_zlim(mid_z - max_range, mid_z + max_range)
    ax_xyz.set_title(f"Trajectoire 3D (T_opt = {T_opt:.2f}s)")

    # Profils Vitesse / Accélération
    ax_v.plot(t_vals, vel_arr, 'g-', alpha=0.3)
    vel_line, = ax_v.plot([], [], 'g-', linewidth=2.5)
    ax_v.axhline(V_MAX_ROBOT, color='r', linestyle='--', label='V_max')
    ax_v.set_ylim(0, V_MAX_ROBOT * 1.1)
    ax_v.set_title('Profil de vitesse (Norme)')
    ax_v.legend()

    ax_a.plot(t_vals, acc_arr, 'm-', alpha=0.3)
    acc_line, = ax_a.plot([], [], 'm-', linewidth=2.5)
    ax_a.axhline(A_MAX_ROBOT, color='r', linestyle='--', label='A_max')
    ax_a.axhline(-A_MAX_ROBOT, color='orange', linestyle='--')
    ax_a.set_ylim(-A_MAX_ROBOT * 1.15, A_MAX_ROBOT * 1.15)
    ax_a.set_title('Profil d\'accélération tangentielle')
    ax_a.legend()

    def update(frame):
        t = t_vals[frame]
        p = pos_arr[frame]

        traj_line.set_data(pos_arr[:frame+1, 0], pos_arr[:frame+1, 1])
        traj_line.set_3d_properties(pos_arr[:frame+1, 2])
        
        robot_dot.set_data([p[0]], [p[1]])
        robot_dot.set_3d_properties([p[2]])

        vel_line.set_data(t_vals[:frame+1], vel_arr[:frame+1])
        acc_line.set_data(t_vals[:frame+1], acc_arr[:frame+1])
        
        return traj_line, robot_dot, vel_line, acc_line

    anim = FuncAnimation(fig, update, frames=n_frames, interval=20, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ctrl = VelocityController()
    
    # robot_state : [x, y, z, dir_x, dir_y, dir_z, v_start]
    robot = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0] 
    # target_state : [x, y, z, dir_x, dir_y, dir_z, v_end]
    target = [-2.0, -3.0, -4.0, 0.0, 0.0, 1.0, 0.5] 
    
    run_demo(ctrl, robot, target)