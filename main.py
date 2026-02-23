import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import pandas as pd

dt = 1.0
steps = 10

t = 0.0
x = 0.0
v = 0.0

a0 = 2.0
k = 0.2

data = []

for _ in range(steps + 1):
    a = a0 + k * t

    data.append({
        "t": t,
        "x": x,
        "v": v,
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