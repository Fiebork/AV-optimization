import numpy as np

def normalize_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

def find_target_index(x, y, x_path, y_path, lookahead_distance):
    distances = np.hypot(x_path - x, y_path - y)
    nearest_index = int(np.argmin(distances))
    target_index = nearest_index
    traveled = 0.0
    path_count = len(x_path)
    while target_index < path_count - 1 and traveled < lookahead_distance:
        dx = x_path[target_index + 1] - x_path[target_index]
        dy = y_path[target_index + 1] - y_path[target_index]
        traveled += float(np.hypot(dx, dy))
        target_index += 1
    return nearest_index, target_index, float(traveled)

def pure_pursuit_steering(x, y, theta, v, x_path, y_path, wheelbase, k_lookahead, min_lookahead, max_steer):
    lookahead_distance = k_lookahead * v + min_lookahead
    nearest_index, target_index, actual_lookahead = find_target_index(
        x, y, x_path, y_path, lookahead_distance
    )
    tx = x_path[target_index]
    ty = y_path[target_index]
    alpha = normalize_angle(np.arctan2(ty - y, tx - x) - theta)
    delta = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead_distance)
    delta = float(np.clip(delta, -max_steer, max_steer))
    return delta, nearest_index, target_index, float(lookahead_distance), actual_lookahead

def speed_control(v, target_speed, kp_speed, min_accel, max_accel):
    a = kp_speed * (target_speed - v)
    return float(np.clip(a, min_accel, max_accel))

def propagate_kinematic_bicycle(x, y, theta, v, delta, a, wheelbase, dt):
    vx = float(v * np.cos(theta))
    vy = float(v * np.sin(theta))
    yaw_rate = float(v * np.tan(delta) / wheelbase)
    x_next = x + vx * dt
    y_next = y + vy * dt
    theta_next = normalize_angle(theta + yaw_rate * dt)
    v_next = v + a * dt
    return x_next, y_next, theta_next, v_next, vx, vy, yaw_rate

def has_passed_path_end(x, y, x_path, y_path, end_stop_distance):
    if len(x_path) < 2:
        return False
    end_x = float(x_path[-1])
    end_y = float(y_path[-1])
    tx = float(x_path[-1] - x_path[-2])
    ty = float(y_path[-1] - y_path[-2])
    tangent_norm = float(np.hypot(tx, ty))
    if tangent_norm <= 1e-9:
        return False
    tx /= tangent_norm
    ty /= tangent_norm
    rel_x = float(x - end_x)
    rel_y = float(y - end_y)
    along_track = rel_x * tx + rel_y * ty
    distance_to_end = float(np.hypot(rel_x, rel_y))
    return along_track > 0.0 and distance_to_end <= end_stop_distance

def distance_to_path_end(x, y, x_path, y_path):
    return float(np.hypot(x - x_path[-1], y - y_path[-1]))