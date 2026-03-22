import numpy as np
import pandas as pd

from car import (
    build_path_s,
    has_passed_path_end,
    idm_acceleration,
    interpolate_path_pose,
    project_to_path_s,
    propagate_kinematic_bicycle,
    pure_pursuit_steering,
)


def simulate_traffic(
    x_path,
    y_path,
    dt,
    steps,
    wheelbase,
    k_lookahead,
    min_lookahead,
    max_steer,
    min_accel,
    max_accel_clip,
    idm_max_accel,
    num_cars=5,
    follower_desired_speed=15.0,
    leader_desired_speed=9.0,
    time_headway=1.2,
    min_spacing=2.0,
    accel_exponent=4.0,
    comfortable_brake=2.0,
    initial_spacing=12.0,
    leader_start_s=60.0,
    vehicle_length=4.5,
    end_stop_distance=10.0,
):
    x_path = np.asarray(x_path, dtype=float)
    y_path = np.asarray(y_path, dtype=float)
    s_path = build_path_s(x_path, y_path)

    desired_speed = np.full(num_cars, float(follower_desired_speed), dtype=float)
    desired_speed[0] = float(leader_desired_speed)

    s_initial = np.array([leader_start_s - i * initial_spacing for i in range(num_cars)], dtype=float)
    if np.min(s_initial) < 0.0:
        s_initial += -float(np.min(s_initial)) + 1.0

    x = np.zeros(num_cars, dtype=float)
    y = np.zeros(num_cars, dtype=float)
    theta = np.zeros(num_cars, dtype=float)
    v = np.zeros(num_cars, dtype=float)
    for i in range(num_cars):
        x[i], y[i], theta[i] = interpolate_path_pose(x_path, y_path, s_path, s_initial[i])

    t = 0.0
    data = []
    active = np.ones(num_cars, dtype=bool)
    dt_sq = max(dt * dt, 1e-9)
    emergency_overlap_margin = 0.1

    for _ in range(steps + 1):
        if not np.any(active):
            print(f"All vehicles passed the path end at t={t:.2f} s")
            break

        deltas = np.zeros(num_cars, dtype=float)
        along_s = np.full(num_cars, fill_value=np.nan, dtype=float)

        for i in range(num_cars):
            if not active[i]:
                continue
            delta, _, _, _, _ = pure_pursuit_steering(
                x[i],
                y[i],
                theta[i],
                v[i],
                x_path,
                y_path,
                wheelbase=wheelbase,
                k_lookahead=k_lookahead,
                min_lookahead=min_lookahead,
                max_steer=max_steer,
            )
            deltas[i] = delta
            along_s[i] = project_to_path_s(
                x[i],
                y[i],
                x_path,
                y_path,
                s_path=s_path,
            )

        active_indices = np.where(active)[0]
        order = active_indices[np.argsort(along_s[active_indices])[::-1]]
        leader_for = np.full(num_cars, fill_value=-1, dtype=int)
        for rank in range(1, len(order)):
            follower = int(order[rank])
            leader_for[follower] = int(order[rank - 1])

        accelerations = np.zeros(num_cars, dtype=float)
        for i in range(num_cars):
            if not active[i]:
                continue
            leader_idx = leader_for[i]
            obstacle_gap = None
            obstacle_speed = 0.0
            if leader_idx >= 0:
                gap_to_leader = along_s[leader_idx] - along_s[i] - vehicle_length
                obstacle_gap = float(max(1e-3, gap_to_leader))
                obstacle_speed = float(v[leader_idx])

            approach_rate = float(v[i] - obstacle_speed)
            accelerations[i] = idm_acceleration(
                v=v[i],
                target_speed=desired_speed[i],
                gap=obstacle_gap,
                approach_rate=approach_rate,
                idm_max_accel=idm_max_accel,
                comfortable_brake=comfortable_brake,
                time_headway=time_headway,
                min_spacing=min_spacing,
                accel_exponent=accel_exponent,
                min_accel=min_accel,
                max_accel_clip=max_accel_clip,
            )

        predicted_s = np.full(num_cars, fill_value=np.nan, dtype=float)
        for i in active_indices:
            i = int(i)
            predicted_s[i] = along_s[i] + v[i] * dt + 0.5 * accelerations[i] * dt_sq

        min_physical_spacing = float(vehicle_length + emergency_overlap_margin)
        for idx in order[1:]:
            follower = int(idx)
            leader_idx = int(leader_for[follower])
            if leader_idx < 0:
                continue
            next_gap = float(predicted_s[leader_idx] - predicted_s[follower])
            if next_gap >= min_physical_spacing:
                continue

            emergency_accel_limit = 2.0 * (
                predicted_s[leader_idx] - min_physical_spacing - along_s[follower] - v[follower] * dt
            ) / dt_sq
            accelerations[follower] = float(
                np.clip(min(accelerations[follower], emergency_accel_limit), min_accel, max_accel_clip)
            )
            predicted_s[follower] = along_s[follower] + v[follower] * dt + 0.5 * accelerations[follower] * dt_sq

        x_next = x.copy()
        y_next = y.copy()
        theta_next = theta.copy()
        v_next = v.copy()

        for i in range(num_cars):
            if not active[i]:
                continue
            yaw_rate = float(v[i] * np.tan(deltas[i]) / wheelbase)
            data.append({
                "t": t,
                "car_id": i,
                "x": x[i],
                "y": y[i],
                "speed": float(v[i]),
                "target_speed": float(desired_speed[i]),
                "delta": float(deltas[i]),
                "ax": float(accelerations[i]),
                "yaw_rate": yaw_rate,
                "theta": float(theta[i]),
                "s": float(along_s[i]),
            })

            xn, yn, thn, vn, _, _, _ = propagate_kinematic_bicycle(
                x=x[i],
                y=y[i],
                theta=theta[i],
                v=v[i],
                delta=deltas[i],
                a=accelerations[i],
                wheelbase=wheelbase,
                dt=dt,
            )
            x_next[i] = xn
            y_next[i] = yn
            theta_next[i] = thn
            v_next[i] = vn

        for i in range(num_cars):
            if not active[i]:
                continue
            passed_end = has_passed_path_end(
                x_next[i],
                y_next[i],
                x_path,
                y_path,
                end_stop_distance=end_stop_distance,
            )
            if passed_end:
                active[i] = False
                x_next[i] = np.nan
                y_next[i] = np.nan
                theta_next[i] = np.nan
                v_next[i] = 0.0

        x = x_next
        y = y_next
        theta = theta_next
        v = v_next
        t += dt

    return pd.DataFrame(data)
