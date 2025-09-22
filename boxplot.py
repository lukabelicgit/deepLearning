import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("exercise_data.csv")

def calculate_total_time(row):
    if isinstance(row['Time Between Reps'], str):
        times = map(float, row['Time Between Reps'].split(','))
        return sum(times)
    return float(row['Time Between Reps'])

df['Total Time'] = df.apply(calculate_total_time, axis=1)

plt.figure(figsize=(12, 6))
sns.boxplot(x='Exercise', y='Total Time', data=df)
plt.xlabel('Exercises')
plt.ylabel('Total Time Spent (seconds)')
plt.title('Distribution of Total Time Spent by Exercise')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()