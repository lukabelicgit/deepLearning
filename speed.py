import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("exercise_data.csv")

def parse_times(s):
    s = str(s).strip()
    if not s:
        return []
    return [float(x) for x in s.split(",") if x != ""]

def safe_avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")

df["Time Between Reps"] = df["Time Between Reps"].apply(parse_times)
df["Average Time Per Rep"] = df["Time Between Reps"].apply(safe_avg)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Per-exercise cumulative reps (more meaningful x-axis per chart)
df["Cum Reps (per exercise)"] = df.groupby("Exercise")["Reps"].cumsum()

exercises = df["Exercise"].unique()
fig, axes = plt.subplots(len(exercises), 1, figsize=(10, 6 * max(1, len(exercises))))

if len(exercises) == 1:
    axes = [axes]

for ax, exercise in zip(axes, exercises):
    sub = df[df["Exercise"] == exercise]
    ax.plot(sub["Cum Reps (per exercise)"], sub["Average Time Per Rep"], marker="o", linestyle="-")
    ax.set_title(f"Average Time Per Rep vs Cumulative Reps ({exercise.capitalize()})")
    ax.set_xlabel("Cumulative Reps (this exercise)")
    ax.set_ylabel("Average Time Per Rep (seconds)")
    ax.grid(True)

plt.tight_layout(pad=5.0)
plt.show()
