
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import math
from matplotlib.animation import FuncAnimation

class RobotTrajectoryPlanner:
    def __init__(self):
        # Thông số robot - CẬP NHẬT
        self.robot_max_speed = 5.0  # m/s
        self.robot_acceleration = 3.0  # m/s²
        self.robot_radius = 0.2  # m
        self.max_rotation_speed = 1.5  # rad/s
        
        # Thông số B-Spline
        self.degree = 3  # Bậc 3 cho đường cong mượt
        
    def generate_control_points(self, start_pos, start_theta, target_pos, target_theta):
        """
        Tạo các điểm kiểm soát cho B-Spline từ start đến target
        """
        control_points = []
        
        # Điểm 1: Vị trí bắt đầu
        control_points.append(start_pos)
        
        # Điểm 2: Điểm định hướng từ start (dựa trên hướng ban đầu)
        start_dir = np.array([math.cos(start_theta), math.sin(start_theta)])
        mid1 = start_pos + start_dir * 2.0
        control_points.append(mid1)
        
        # Điểm 3: Điểm trung gian (tạo đường cong)
        mid2 = (start_pos + target_pos) / 2
        direction = target_pos - start_pos
        direction_norm = direction / np.linalg.norm(direction)
        perpendicular = np.array([-direction_norm[1], direction_norm[0]]) * 3.0
        mid2 = mid2 + perpendicular
        control_points.append(mid2)
        
        # Điểm 4: Điểm định hướng đến target
        target_dir = np.array([math.cos(target_theta), math.sin(target_theta)])
        mid3 = target_pos - target_dir * 2.0
        control_points.append(mid3)
        
        # Điểm 5: Vị trí đích
        control_points.append(target_pos)
        
        return np.array(control_points)
    
    def create_bspline_trajectory(self, control_points, num_samples=150):
        """
        Tạo quỹ đạo B-Spline từ các điểm kiểm soát
        """
        n = len(control_points)
        
        knots = np.zeros(n + self.degree + 1)
        for i in range(self.degree + 1):
            knots[i] = 0
            knots[n + i] = 1
            
        if n > self.degree + 1:
            interior_knots = np.linspace(0, 1, n - self.degree + 1)
            knots[self.degree+1:n] = interior_knots[1:-1]
        
        t = np.linspace(0, 1, num_samples)
        spline_x = BSpline(knots, control_points[:, 0], self.degree)
        spline_y = BSpline(knots, control_points[:, 1], self.degree)
        
        trajectory = np.column_stack([spline_x(t), spline_y(t)])
        
        return trajectory, t
    
    def calculate_orientation_profile(self, trajectory, start_theta, target_theta):
        """
        Tính toán profile hướng cho robot dọc theo quỹ đạo
        """
        orientations = []
        
        for i, pos in enumerate(trajectory):
            progress = i / (len(trajectory) - 1)
            
            if i < len(trajectory) - 1:
                dx = trajectory[i+1, 0] - pos[0]
                dy = trajectory[i+1, 1] - pos[1]
                tangent_theta = math.atan2(dy, dx)
                
                if progress < 0.8:
                    theta = tangent_theta
                else:
                    blend = (progress - 0.8) / 0.2
                    theta = tangent_theta * (1 - blend) + target_theta * blend
            else:
                theta = target_theta
            
            orientations.append(theta)
        
        orientations[0] = start_theta
        orientations[-1] = target_theta
        
        return orientations
    
    def calculate_velocity_profile(self, trajectory, max_speed=5.0):
        """
        Tính toán profile vận tốc dọc theo quỹ đạo
        """
        velocities = []
        current_speed = 0
        
        for i in range(len(trajectory)):
            if i == 0:
                velocities.append(0.0)
                current_speed = 0
            else:
                progress = i / len(trajectory)
                
                if progress < 0.3:  # Tăng tốc
                    current_speed = min(max_speed, current_speed + self.robot_acceleration * 0.1)
                elif progress > 0.9:  # Giảm tốc trong 10% cuối
                    # Giảm tốc để về 0 tại điểm cuối
                    remaining_progress = 1.0 - progress
                    current_speed = max_speed * (remaining_progress / 0.1)
                    # Đảm bảo không âm
                    current_speed = max(0, current_speed)
                else:  # Duy trì tốc độ
                    current_speed = max_speed
                
                velocities.append(current_speed)
        
        # Đảm bảo điểm cuối cùng có vận tốc 0
        velocities[-1] = 0.0
        
        return velocities
    
    def calculate_angular_velocity_profile(self, orientations, time_step=0.1):
        """
        Tính toán profile vận tốc góc với giới hạn 1.5 rad/s
        """
        angular_velocities = [0.0]
        
        for i in range(1, len(orientations)):
            delta_theta = orientations[i] - orientations[i-1]
            
            while delta_theta > math.pi:
                delta_theta -= 2 * math.pi
            while delta_theta < -math.pi:
                delta_theta += 2 * math.pi
                
            angular_vel = delta_theta / time_step
            angular_vel = max(min(angular_vel, self.max_rotation_speed), -self.max_rotation_speed)
            angular_velocities.append(angular_vel)
        
        return angular_velocities
    
    def real_time_trajectory_simulation(self, start_pos, start_theta, target_pos, target_theta):
        """
        Mô phỏng di chuyển với hiển thị thời gian thực trên 2 biểu đồ riêng
        """
        # Tạo điểm kiểm soát
        control_points = self.generate_control_points(start_pos, start_theta, target_pos, target_theta)
        
        # Tạo quỹ đạo B-Spline
        trajectory, times = self.create_bspline_trajectory(control_points)
        
        # Tính profile hướng
        orientations = self.calculate_orientation_profile(trajectory, start_theta, target_theta)
        
        # Tính profile vận tốc
        velocities = self.calculate_velocity_profile(trajectory)
        
        # Tính profile vận tốc góc
        angular_velocities = self.calculate_angular_velocity_profile(orientations)
        
        # Tạo figure với 3 subplot
        fig = plt.figure(figsize=(16, 10))
        
        # Subplot 1: Quỹ đạo
        ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
        ax1.set_xlim(-2, 12)
        ax1.set_ylim(-2, 10)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('B-SPLINE')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Subplot 2: Biểu đồ vận tốc
        ax2 = plt.subplot2grid((3, 2), (0, 1))
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity over time')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.5, self.robot_max_speed + 0.5)
        
        # Subplot 3: Biểu đồ góc
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Theta (rad)')
        ax3.set_title('Theta over time')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-3.5, 3.5)
        
        # Subplot 4: Biểu đồ vận tốc góc
        ax4 = plt.subplot2grid((3, 2), (2, 1))
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Omega (rad/s)')
        ax4.set_title('Omega over time')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-self.max_rotation_speed - 0.5, self.max_rotation_speed + 0.5)
        
        # Vẽ quỹ đạo hoàn chỉnh
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3, linewidth=2, label='Quỹ đạo B-Spline')
        ax1.plot(control_points[:, 0], control_points[:, 1], 'ro--', alpha=0.6, label='Điểm kiểm soát')
        
        # Vẽ điểm bắt đầu và kết thúc
        ax1.plot(start_pos[0], start_pos[1], 'go', markersize=8, label='Start')
        ax1.plot(target_pos[0], target_pos[1], 'rx', markersize=10, label='Target')
        
        # Vẽ hướng start và target
        start_arrow_len = 0.8
        start_dx = start_arrow_len * math.cos(start_theta)
        start_dy = start_arrow_len * math.sin(start_theta)
        ax1.arrow(start_pos[0], start_pos[1], start_dx, start_dy, 
                 head_width=0.1, head_length=0.2, fc='green', ec='green')
        
        target_arrow_len = 0.8
        target_dx = target_arrow_len * math.cos(target_theta)
        target_dy = target_arrow_len * math.sin(target_theta)
        ax1.arrow(target_pos[0], target_pos[1], target_dx, target_dy, 
                 head_width=0.1, head_length=0.2, fc='red', ec='red')
        
        ax1.legend()
        
        # Khởi tạo các đối tượng đồ họa cho animation
        robot_circle = plt.Circle(start_pos, self.robot_radius, color='blue', alpha=0.7)
        ax1.add_patch(robot_circle)
        
        robot_arrow = ax1.arrow(start_pos[0], start_pos[1], 
                              0.5 * math.cos(start_theta), 0.5 * math.sin(start_theta),
                              head_width=0.1, head_length=0.15, fc='black', ec='black')
        
        # Khởi tạo biểu đồ vận tốc
        time_axis = np.arange(len(trajectory)) * 0.1
        velocity_line, = ax2.plot([], [], 'b-', linewidth=2, label='Velocity')
        ax2.axhline(y=self.robot_max_speed, color='r', linestyle='--', alpha=0.7, 
                   label=f'Limited: {self.robot_max_speed} m/s')
        ax2.legend()
        
        # Khởi tạo biểu đồ góc
        orientation_line, = ax3.plot([], [], 'g-', linewidth=2, label='Góc')
        ax3.legend()
        
        # Khởi tạo biểu đồ vận tốc góc
        angular_velocity_line, = ax4.plot([], [], 'r-', linewidth=2, label='Omega')
        ax4.axhline(y=self.max_rotation_speed, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Limited: ±{self.max_rotation_speed} rad/s')
        ax4.axhline(y=-self.max_rotation_speed, color='orange', linestyle='--', alpha=0.7)
        ax4.legend()
        
        # Biến để theo dõi thời gian
        current_time = 0
        
        # Text hiển thị thông tin
        info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def animate(frame):
            nonlocal current_time, robot_arrow
            
            current_time = frame * 0.1
            
            robot_idx = min(frame, len(trajectory) - 1)
            
            # Cập nhật vị trí và hướng robot
            robot_pos_current = trajectory[robot_idx]
            robot_theta_current = orientations[robot_idx]
            
            # Cập nhật đồ họa robot
            robot_circle.center = robot_pos_current
            
            # Cập nhật mũi tên hướng robot
            robot_arrow.remove()
            arrow_length = 0.6
            dx = arrow_length * math.cos(robot_theta_current)
            dy = arrow_length * math.sin(robot_theta_current)
            robot_arrow = ax1.arrow(robot_pos_current[0], robot_pos_current[1],
                                  dx, dy,
                                  head_width=0.1, head_length=0.15, fc='black', ec='black')
            
            # CẬP NHẬT BIỂU ĐỒ THEO THỜI GIAN THỰC
            # Biểu đồ vận tốc
            velocity_line.set_data(time_axis[:robot_idx+1], velocities[:robot_idx+1])
            ax2.set_xlim(0, max(current_time + 0.5, 1))  # Tự động điều chỉnh trục x
            
            # Biểu đồ góc
            orientation_line.set_data(time_axis[:robot_idx+1], orientations[:robot_idx+1])
            ax3.set_xlim(0, max(current_time + 0.5, 1))
            
            # Biểu đồ vận tốc góc
            angular_velocity_line.set_data(time_axis[:robot_idx+1], angular_velocities[:robot_idx+1])
            ax4.set_xlim(0, max(current_time + 0.5, 1))
            
            # Cập nhật thông tin
            info_text.set_text(f'Times: {current_time:.1f}s\n'
                             f'Position: ({robot_pos_current[0]:.2f}, {robot_pos_current[1]:.2f})\n'
                             f'Theta: {math.degrees(robot_theta_current):.1f}°\n'
                             f'V: {velocities[robot_idx]:.2f} m/s\n'
                             f'Omega: {angular_velocities[robot_idx]:.2f} rad/s')
            
            return (robot_circle, robot_arrow, velocity_line, orientation_line, 
                   angular_velocity_line, info_text)
        
        # Tạo animation
        total_frames = len(trajectory)
        anim = FuncAnimation(fig, animate, frames=total_frames, 
                           interval=100, blit=False, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim, trajectory, orientations, velocities, angular_velocities

# Demo robot di chuyển
def demo_robot_trajectory():
    planner = RobotTrajectoryPlanner()
    
    # Điểm bắt đầu và kết thúc
    start_position = np.array([1.0, 1.0])
    start_theta = math.radians(30)
    
    target_position = np.array([9.0, 8.0])
    target_theta = math.radians(-45)
    
    print("=== DEMO DI CHUYỂN ROBOT - HIỂN THỊ THỜI GIAN THỰC ===")
    print(f"Vmax: {planner.robot_max_speed} m/s")
    print(f"Omega: {planner.max_rotation_speed} rad/s")
    print(f"Start: ({start_position[0]}, {start_position[1]}), góc: {math.degrees(start_theta):.1f}°")
    print(f"Targer: ({target_position[0]}, {target_position[1]}), góc: {math.degrees(target_theta):.1f}°")
    
    # Chạy mô phỏng
    anim, trajectory, orientations, velocities, angular_velocities = planner.real_time_trajectory_simulation(
        start_position, start_theta, target_position, target_theta
    )
    
    # Hiển thị thống kê sau khi mô phỏng
    print("\n=== THỐNG KÊ CHUYỂN ĐỘNG ===")
    print(f"Time: {len(trajectory) * 0.1:.1f} giây")
    print(f"Quãng đường: {np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)):.2f} mét")
    print(f"V_avg: {np.mean(velocities):.2f} m/s")
    print(f"Vmax: {np.max(velocities):.2f} m/s")
    print(f"Omega_max: {np.max(np.abs(angular_velocities)):.2f} rad/s")
    
    return planner, anim, trajectory, orientations, velocities, angular_velocities

# Chạy demo
if __name__ == "__main__":
    planner, anim, trajectory, orientations, velocities, angular_velocities = demo_robot_trajectory()