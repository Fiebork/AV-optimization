import numpy as np

def straight_road(length=1000.0, step=1.0, y=0.0):
    x_path = np.arange(0.0, length + step, step, dtype=float)
    y_path = np.full_like(x_path, fill_value=float(y), dtype=float)
    return x_path, y_path

def straight_with_curve(
    entry_length=80.0,
    radius=40.0,
    arc_angle_deg=60.0,
    exit_length=80.0,
    step=1.0,
):
    entry_x = np.arange(0.0, entry_length + step, step, dtype=float)
    entry_y = np.zeros_like(entry_x, dtype=float)

    arc_angle = np.deg2rad(arc_angle_deg)
    dphi = step / radius
    phi = np.arange(-np.pi / 2.0, -np.pi / 2.0 + arc_angle + dphi, dphi, dtype=float)
    center_x = entry_length
    center_y = radius
    arc_x = center_x + radius * np.cos(phi)
    arc_y = center_y + radius * np.sin(phi)

    end_x = float(arc_x[-1])
    end_y = float(arc_y[-1])
    end_heading = float(arc_angle)
    s_exit = np.arange(step, exit_length + step, step, dtype=float)
    exit_x = end_x + s_exit * np.cos(end_heading)
    exit_y = end_y + s_exit * np.sin(end_heading)

    x_path = np.concatenate((entry_x, arc_x[1:], exit_x))
    y_path = np.concatenate((entry_y, arc_y[1:], exit_y))
    return x_path, y_path

def right_angle_turn(
    entry_length=120.0,
    radius=40.0,
    exit_length=120.0,
    step=1.0,
):
    entry_x = np.arange(0.0, entry_length + step, step, dtype=float)
    entry_y = np.zeros_like(entry_x, dtype=float)

    dphi = step / radius
    phi = np.arange(-0.5 * np.pi, 0.0 + dphi, dphi, dtype=float)
    center_x = entry_length
    center_y = radius
    arc_x = center_x + radius * np.cos(phi)
    arc_y = center_y + radius * np.sin(phi)

    end_x = float(arc_x[-1])
    end_y = float(arc_y[-1])
    s_exit = np.arange(step, exit_length + step, step, dtype=float)
    exit_x = np.full_like(s_exit, fill_value=end_x, dtype=float)
    exit_y = end_y + s_exit

    x_path = np.concatenate((entry_x, arc_x[1:], exit_x))
    y_path = np.concatenate((entry_y, arc_y[1:], exit_y))
    return x_path, y_path

def road_edges(x_path, y_path, road_width=8.0):
    x_path = np.asarray(x_path, dtype=float)
    y_path = np.asarray(y_path, dtype=float)
    if x_path.size < 2:
        return x_path.copy(), y_path.copy(), x_path.copy(), y_path.copy()

    dx = np.gradient(x_path)
    dy = np.gradient(y_path)
    norm = np.hypot(dx, dy)
    norm = np.where(norm > 1e-9, norm, 1.0)
    nx = -dy / norm
    ny = dx / norm
    half_width = 0.5 * float(road_width)

    left_x = x_path + half_width * nx
    left_y = y_path + half_width * ny
    right_x = x_path - half_width * nx
    right_y = y_path - half_width * ny
    return left_x, left_y, right_x, right_y