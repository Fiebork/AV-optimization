import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from road import road_edges

def animate_position(df, fps=60, speed=5.0, path_x=None, path_y=None, road_width=8.0):
    x_values = df["x"].to_numpy()
    y_values = df["y"].to_numpy()
    t_values = df["t"].to_numpy()
    theta_values = df["theta"].to_numpy()

    sim_dt = float(t_values[1] - t_values[0])
    step_per_frame = max(1, int(round(speed / (fps * sim_dt))))
    frame_indices = range(0, len(t_values), step_per_frame)

    fig, ax = plt.subplots()

    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    if path_x is not None and path_y is not None:
        left_x, left_y, right_x, right_y = road_edges(path_x, path_y, road_width=road_width)
        x_min = min(x_min, float(np.min(left_x)), float(np.min(right_x)))
        x_max = max(x_max, float(np.max(left_x)), float(np.max(right_x)))
        y_min = min(y_min, float(np.min(left_y)), float(np.min(right_y)))
        y_max = max(y_max, float(np.max(left_y)), float(np.max(right_y)))
    margin_x = max(1.0, 0.1 * (x_max - x_min if x_max > x_min else 1.0))
    margin_y = max(10.0, 0.2 * (y_max - y_min if y_max > y_min else 1.0))

    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Car Animation")
    ax.set_aspect("auto")
    ax.grid(True)

    if path_x is not None and path_y is not None:
        polygon_x = np.concatenate((left_x, right_x[::-1]))
        polygon_y = np.concatenate((left_y, right_y[::-1]))
        ax.fill(polygon_x, polygon_y, color="lightsteelblue", alpha=0.6, label="road")
    ax.plot(x_values, y_values, color="lightgray", linewidth=1.0, label="trajectory")
    (point,) = ax.plot([x_values[0]], [y_values[0]], "ro", markersize=8)
    heading_length = 3.0
    (heading_line,) = ax.plot([], [], "r-", linewidth=2.0)
    time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)
    ax.legend()

    def init():
        x0 = x_values[0]
        y0 = y_values[0]
        th0 = theta_values[0]
        point.set_data([x0], [y0])
        heading_line.set_data(
            [x0, x0 + heading_length * np.cos(th0)],
            [y0, y0 + heading_length * np.sin(th0)],
        )
        time_text.set_text(f"t = {t_values[0]:.2f} s")
        return point, heading_line, time_text

    def update(i):
        x = x_values[i]
        y = y_values[i]
        th = theta_values[i]
        point.set_data([x], [y])
        heading_line.set_data(
            [x, x + heading_length * np.cos(th)],
            [y, y + heading_length * np.sin(th)],
        )
        time_text.set_text(f"t = {t_values[i]:.2f} s")
        return point, heading_line, time_text

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    plt.show(block=True)
    return animation