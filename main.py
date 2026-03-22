import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from animation import animate_platoon_position
from road import right_angle_turn, straight_road, straight_with_curve
from simulation import simulate_traffic


SIMULATION = {
    "dt": 0.1,
    "wheelbase": 2.8,
    "k_lookahead": 0.1,
    "min_lookahead": 2.0,
    "max_steer": np.deg2rad(30.0),
    "max_accel_clip": 2.0,
    "min_accel": -4.0,
    "road_width": 8.0,
}

TRAFFIC = {
    "num_cars": 5,
    "steps": 1400,
    "animation_fps": 60,
    "animation_speed": 6.0,
    "follower_desired_speed": 15.0,
    "leader_desired_speed": 9.0,
    "time_headway": 1.2,
    "min_spacing": 2.0,
    "accel_exponent": 4.0,
    "comfortable_brake": 2.0,
    "initial_spacing": 12.0,
    "leader_start_s": 60.0,
    "vehicle_length": 4.5,
    "end_stop_distance": 10.0,
    "idm_max_accel": 2.0,
}


def build_scenarios():
    return {
        "1": {
            "name": "Straight Road",
            "path_builder": lambda: straight_road(length=300.0, step=1.0, y=0.0),
            "traffic_overrides": {
                "num_cars": 1,
                "steps": 1000,
                "leader_desired_speed": 15.0,
            },
        },
        "2": {
            "name": "Curved Road",
            "path_builder": lambda: straight_with_curve(
                entry_length=80.0,
                radius=40.0,
                arc_angle_deg=60.0,
                exit_length=100.0,
                step=1.0,
            ),
            "traffic_overrides": {
                "num_cars": 1,
                "steps": 1000,
                "leader_desired_speed": 15.0,
            },
        },
        "3": {
            "name": "Right Angle Turn",
            "path_builder": lambda: right_angle_turn(
                entry_length=120.0,
                radius=40.0,
                exit_length=120.0,
                step=1.0,
            ),
            "traffic_overrides": {
                "num_cars": 1,
                "steps": 1000,
                "leader_desired_speed": 15.0,
            },
        },
        "4": {
            "name": "Curved Traffic Flow",
            "path_builder": lambda: straight_with_curve(
                entry_length=80.0,
                radius=40.0,
                arc_angle_deg=60.0,
                exit_length=140.0,
                step=1.0,
            ),
            "traffic_overrides": {},
        },
        "5": {
            "name": "Right Angle Traffic Flow",
            "path_builder": lambda: right_angle_turn(
                entry_length=120.0,
                radius=40.0,
                exit_length=120.0,
                step=1.0,
            ),
            "traffic_overrides": {
                "num_cars": 5,
                "steps": 1400,
                "leader_desired_speed": 9.0,
            },
        },
        "6": {
            "name": "Slow Leader Adaptation",
            "path_builder": lambda: straight_with_curve(
                entry_length=80.0,
                radius=40.0,
                arc_angle_deg=60.0,
                exit_length=500.0,
                step=1.0,
            ),
            "traffic_overrides": {
                "steps": 1300,
                "follower_desired_speed": 16.0,
                "leader_desired_speed": 6.5,
                "time_headway": 1.7,
                "min_spacing": 2.5,
                "comfortable_brake": 2.5,
                "initial_spacing": 10.0,
                "leader_start_s": 50.0,
            },
        },
    }


def choose_scenario(scenarios):
    print("Available scenarios:")
    for key, scenario in scenarios.items():
        print(f"  {key}: {scenario['name']}")

    try:
        choice = input("Select scenario to run (number): ").strip()
    except EOFError:
        choice = "1"

    if choice not in scenarios:
        print("Invalid selection, defaulting to scenario 1.")
        choice = "1"

    scenario = scenarios[choice]
    print(f"Running scenario {choice}: {scenario['name']}")
    return scenario


def merge_config(base, overrides=None):
    merged = dict(base)
    if overrides:
        merged.update(overrides)
    return merged


def plot_speed_profiles(df):
    plt.figure()
    for car_id, car_data in df.groupby("car_id"):
        plt.plot(car_data["t"], car_data["speed"], label=f"car {car_id}")
    plt.xlabel("t")
    plt.ylabel("speed")
    plt.title("Speed Profiles")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_state_profiles(df):
    plt.figure()
    for car_id, car_data in df.groupby("car_id"):
        plt.plot(car_data["t"], car_data["x"], label=f"car {car_id}")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("x(t)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    for car_id, car_data in df.groupby("car_id"):
        plt.plot(car_data["t"], car_data["y"], label=f"car {car_id}")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("y(t)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    for car_id, car_data in df.groupby("car_id"):
        plt.plot(car_data["t"], car_data["theta"], label=f"car {car_id}")
    plt.xlabel("t")
    plt.ylabel("theta [rad]")
    plt.title("Heading")
    plt.grid(True)
    plt.legend()
    plt.show()


def run_simulation(scenario, sim_cfg, traffic_cfg):
    x_path, y_path = scenario["path_builder"]()
    df = simulate_traffic(
        x_path=x_path,
        y_path=y_path,
        dt=sim_cfg["dt"],
        steps=traffic_cfg["steps"],
        wheelbase=sim_cfg["wheelbase"],
        k_lookahead=sim_cfg["k_lookahead"],
        min_lookahead=sim_cfg["min_lookahead"],
        max_steer=sim_cfg["max_steer"],
        min_accel=sim_cfg["min_accel"],
        max_accel_clip=sim_cfg["max_accel_clip"],
        idm_max_accel=traffic_cfg["idm_max_accel"],
        num_cars=traffic_cfg["num_cars"],
        follower_desired_speed=traffic_cfg["follower_desired_speed"],
        leader_desired_speed=traffic_cfg["leader_desired_speed"],
        time_headway=traffic_cfg["time_headway"],
        min_spacing=traffic_cfg["min_spacing"],
        accel_exponent=traffic_cfg["accel_exponent"],
        comfortable_brake=traffic_cfg["comfortable_brake"],
        initial_spacing=traffic_cfg["initial_spacing"],
        leader_start_s=traffic_cfg["leader_start_s"],
        vehicle_length=traffic_cfg["vehicle_length"],
        end_stop_distance=traffic_cfg["end_stop_distance"],
    )
    return df, x_path, y_path


def show_results(df, x_path, y_path, sim_cfg, traffic_cfg):
    print(df.head(15))
    animation = animate_platoon_position(
        df=df,
        num_cars=traffic_cfg["num_cars"],
        fps=traffic_cfg["animation_fps"],
        speed=traffic_cfg["animation_speed"],
        path_x=x_path,
        path_y=y_path,
        road_width=sim_cfg["road_width"],
    )
    _ = animation
    plot_state_profiles(df)
    plot_speed_profiles(df)


def main():
    scenarios = build_scenarios()
    scenario = choose_scenario(scenarios)

    sim_cfg = dict(SIMULATION)
    traffic_cfg = merge_config(TRAFFIC, scenario.get("traffic_overrides"))
    df, x_path, y_path = run_simulation(
        scenario=scenario,
        sim_cfg=sim_cfg,
        traffic_cfg=traffic_cfg,
    )
    show_results(
        df=df,
        x_path=x_path,
        y_path=y_path,
        sim_cfg=sim_cfg,
        traffic_cfg=traffic_cfg,
    )


if __name__ == "__main__":
    main()
