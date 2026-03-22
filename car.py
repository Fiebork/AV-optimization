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

def build_path_s(x_path, y_path):
    x_path = np.asarray(x_path, dtype=float)
    y_path = np.asarray(y_path, dtype=float)
    if x_path.size <= 1:
        return np.array([0.0], dtype=float)
    ds = np.hypot(np.diff(x_path), np.diff(y_path))
    return np.concatenate(([0.0], np.cumsum(ds)))

def interpolate_path_pose(x_path, y_path, s_path, s_query):
    x_path = np.asarray(x_path, dtype=float)
    y_path = np.asarray(y_path, dtype=float)
    s_path = np.asarray(s_path, dtype=float)
    if x_path.size == 0:
        return 0.0, 0.0, 0.0
    if x_path.size == 1:
        return float(x_path[0]), float(y_path[0]), 0.0

    s = float(np.clip(s_query, s_path[0], s_path[-1]))
    upper = int(np.searchsorted(s_path, s, side="right"))
    idx = int(np.clip(upper - 1, 0, len(s_path) - 2))
    s0 = s_path[idx]
    s1 = s_path[idx + 1]
    seg_len = float(max(1e-9, s1 - s0))
    ratio = (s - s0) / seg_len

    x = float(x_path[idx] + ratio * (x_path[idx + 1] - x_path[idx]))
    y = float(y_path[idx] + ratio * (y_path[idx + 1] - y_path[idx]))
    theta = float(np.arctan2(y_path[idx + 1] - y_path[idx], x_path[idx + 1] - x_path[idx]))
    return x, y, theta

def project_to_path_s(x, y, x_path, y_path, s_path):
    x_path = np.asarray(x_path, dtype=float)
    y_path = np.asarray(y_path, dtype=float)
    s_path = np.asarray(s_path, dtype=float)
    if x_path.size <= 1:
        return 0.0

    x0 = x_path[:-1]
    y0 = y_path[:-1]
    dx = x_path[1:] - x0
    dy = y_path[1:] - y0
    seg_norm_sq = dx * dx + dy * dy
    valid = seg_norm_sq > 1e-12

    rel_x = float(x) - x0
    rel_y = float(y) - y0
    t = np.zeros_like(seg_norm_sq, dtype=float)
    t[valid] = (rel_x[valid] * dx[valid] + rel_y[valid] * dy[valid]) / seg_norm_sq[valid]
    t = np.clip(t, 0.0, 1.0)

    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    dist_sq = (float(x) - proj_x) ** 2 + (float(y) - proj_y) ** 2
    best = int(np.argmin(dist_sq))

    seg_len = float(np.sqrt(max(seg_norm_sq[best], 0.0)))
    return float(s_path[best] + t[best] * seg_len)

def idm_acceleration(
    v,
    target_speed,
    gap,
    approach_rate,
    idm_max_accel,
    comfortable_brake,
    time_headway,
    min_spacing,
    accel_exponent=4.0,
    min_accel=-6.0,
    max_accel_clip=None,
):
    v = float(max(0.0, v))
    target_speed = float(max(1e-3, target_speed))
    idm_max_accel = float(max(1e-3, idm_max_accel))
    comfortable_brake = float(max(1e-3, comfortable_brake))
    time_headway = float(max(0.0, time_headway))
    min_spacing = float(max(0.0, min_spacing))
    accel_exponent = float(max(1.0, accel_exponent))

    free_road_term = (v / target_speed) ** accel_exponent
    if gap is None or not np.isfinite(gap):
        interaction_term = 0.0
    else:
        gap = float(max(1e-3, gap))
        approach_rate = float(approach_rate)
        dynamic_gap = min_spacing + max(
            0.0,
            v * time_headway + (v * approach_rate) / (2.0 * np.sqrt(idm_max_accel * comfortable_brake)),
        )
        interaction_term = (dynamic_gap / gap) ** 2

    accel = idm_max_accel * (1.0 - free_road_term - interaction_term)
    upper = idm_max_accel if max_accel_clip is None else float(max_accel_clip)
    return float(np.clip(accel, float(min_accel), upper))

def propagate_kinematic_bicycle(x, y, theta, v, delta, a, wheelbase, dt):
    vx = float(v * np.cos(theta))
    vy = float(v * np.sin(theta))
    yaw_rate = float(v * np.tan(delta) / wheelbase)
    x_next = x + vx * dt
    y_next = y + vy * dt
    theta_next = normalize_angle(theta + yaw_rate * dt)
    v_next = max(0.0, v + a * dt)
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
