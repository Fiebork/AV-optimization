import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from animation import animate_position

def force_x(t):
    return F0 + amplitude_x * np.sin(omega_x * t)

def force_y(t):
    return amplitude_y * np.cos(omega_y * t)

dt = 0.1
steps = 1000

t = 0.0
x = 0.0
y = 0.0
vx = 0.0
vy = 0.0

m = 1500.0
F0 = 1000.0
amplitude_x = 300.0
omega_x = 0.25
amplitude_y = 1200.0
omega_y = 0.40

data = []

for _ in range(steps + 1):
    Fx = force_x(t)
    Fy = force_y(t)
    ax = Fx / m
    ay = Fy / m
    speed = float(np.hypot(vx, vy))
    theta = float(np.arctan2(vy, vx)) if speed > 1e-9 else 0.0

    data.append({
        "t": t,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "Fx": Fx,
        "Fy": Fy,
        "ax": ax,
        "ay": ay,
        "theta": theta,
    })

    vx += ax * dt
    vy += ay * dt
    x += vx * dt
    y += vy * dt
    t += dt

df = pd.DataFrame(data)
print(df)

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
plt.title("Heading From Velocity")
plt.grid(True)
plt.show()

animation = animate_position(df, fps=60, speed=5.0)