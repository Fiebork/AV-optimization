import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from animation import animate_position
from car import (
    distance_to_path_end,
    has_passed_path_end,
    propagate_kinematic_bicycle,
    pure_pursuit_steering,
    speed_control,
)
from road import right_angle_turn, straight_road, straight_with_curve

def choose_scenario():
    scenarios = {
        "1": ("Straight road", lambda: straight_road(length=300.0, step=1.0, y=0.0)),
        "2": (
            "Curved road",
            lambda: straight_with_curve(
                entry_length=80.0,
                radius=40.0,
                arc_angle_deg=60.0,
                exit_length=100.0,
                step=1.0,
            ),
        ),
        "3": (
            "90-degree turn",
            lambda: right_angle_turn(
                entry_length=120.0,
                radius=40.0,
                exit_length=120.0,
                step=1.0,
            ),
        ),
    }

    print("Available scenarios:")
    for key, (name, _) in scenarios.items():
        print(f"  {key}: {name}")

    try:
        choice = input("Select scenario to run (number): ").strip()
    except EOFError:
        choice = "1"

    if choice not in scenarios:
        print("Invalid selection, defaulting to scenario 1.")
        choice = "1"

    scenario_name, builder = scenarios[choice]
    x_path, y_path = builder()
    print(f"Running scenario {choice}: {scenario_name}")
    return scenario_name, x_path, y_path


dt = 0.1
steps = 1000

t = 0.0
x = 0.0
y = 2.0
theta = 0.0
v = 0.0

L = 2.8
kp_speed = 1.0
target_speed = 15.0
k_lookahead = 0.1
min_lookahead = 2.0
max_steer = np.deg2rad(30.0)
max_accel = 2.0
min_accel = -4.0
road_width = 8.0
end_stop_distance = 10.0
scenario_name, x_path, y_path = choose_scenario()

data = []
last_index = len(x_path) - 1

for _ in range(steps + 1):
    delta, nearest_index, target_index, lookahead_distance, actual_lookahead = pure_pursuit_steering(
        x,
        y,
        theta,
        v,
        x_path,
        y_path,
        wheelbase=L,
        k_lookahead=k_lookahead,
        min_lookahead=min_lookahead,
        max_steer=max_steer,
    )
    a = speed_control(
        v=v,
        target_speed=target_speed,
        kp_speed=kp_speed,
        min_accel=min_accel,
        max_accel=max_accel,
    )

    speed = float(v)
    x_next, y_next, theta_next, v_next, vx, vy, yaw_rate = propagate_kinematic_bicycle(
        x=x,
        y=y,
        theta=theta,
        v=v,
        delta=delta,
        a=a,
        wheelbase=L,
        dt=dt,
    )
    ay = float(v * yaw_rate)

    data.append({
        "t": t,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "target_speed": target_speed,
        "delta": delta,
        "ax": a,
        "ay": ay,
        "yaw_rate": yaw_rate,
        "theta": theta,
    })

    end_distance = distance_to_path_end(x, y, x_path, y_path)
    reached_path_end = nearest_index == last_index and end_distance <= end_stop_distance
    passed_path_end = has_passed_path_end(x, y, x_path, y_path, end_stop_distance=end_stop_distance)
    if target_index == last_index and (reached_path_end or passed_path_end):
        print(f"Car passed end of path at t={t:.2f} s")
        break

    x, y, theta, v = x_next, y_next, theta_next, v_next
    t += dt

df = pd.DataFrame(data)
print(df)

animation = animate_position(
    df,
    fps=60,
    speed=5.0,
    path_x=x_path,
    path_y=y_path,
    road_width=road_width,
)

plt.figure()
plt.plot(df["t"], df["x"])
plt.xlabel("t")
plt.ylabel("x")
plt.title("x(t)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["t"], df["y"])
plt.xlabel("t")
plt.ylabel("y")
plt.title("y(t)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["t"], df["speed"])
plt.xlabel("t")
plt.ylabel("speed")
plt.title("speed(t)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["t"], df["theta"])
plt.xlabel("t")
plt.ylabel("theta [rad]")
plt.title("Heading")
plt.grid(True)
plt.show()