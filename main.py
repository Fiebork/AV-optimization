import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def force(t):
    return F0 + amplitude * np.sin(omega * t)

dt = 1.0
steps = 10

t = 0.0
x = 0.0
v = 0.0

m = 1500.0
F0 = 1000.0
amplitude = 300.0
omega = 0.25

data = []

for _ in range(steps + 1):
    F = force(t)
    a = F / m

    data.append({
        "t": t,
        "x": x,
        "v": v,
        "F": F,
        "a": a
    })

    v += a * dt
    x += v * dt
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
plt.plot(df["t"], df["v"])
plt.xlabel("t")
plt.ylabel("v")
plt.title("v(t)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["t"], df["a"])
plt.xlabel("t")
plt.ylabel("a")
plt.title("a(t)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(df["t"], df["F"])
plt.xlabel("t")
plt.ylabel("F")
plt.title("F(t)")
plt.grid(True)
plt.show()