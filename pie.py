import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("exercise_data.csv")

def calculate_total_time(row):

    if isinstance(row['Time Between Reps'], str):

        times = map(float, row['Time Between Reps'].split(','))
        total_time_seconds = sum(times)
    else:
        total_time_seconds = float(row['Time Between Reps'])

    return total_time_seconds

df['Total Time Seconds'] = df.apply(calculate_total_time, axis=1)

def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

df['Total Time'] = df['Total Time Seconds'].apply(seconds_to_hms)

df['Month'] = pd.to_datetime(df['Date']).dt.month

monthly_time_spent = df.groupby(['Month', 'Exercise'])['Total Time Seconds'].sum().reset_index()

total_time_spent = df.groupby('Exercise')['Total Time Seconds'].sum().reset_index()

month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

num_months = monthly_time_spent['Month'].nunique()
fig, axes = plt.subplots(4, 4, figsize=(16, 12)) 
axes = axes.flatten() 

def custom_pie(ax, data, labels):
    sizes = data['Total Time Seconds']
    total = sizes.sum()
    
    custom_labels = [
        f"{label} \n{seconds_to_hms(time)} ({time/total*100:.1f}%)"
        for label, time in zip(labels, sizes)
    ]
    
    ax.pie(sizes, labels=custom_labels, autopct=None, startangle=140)
    ax.axis('equal') 

for idx, month in enumerate(sorted(monthly_time_spent['Month'].unique())):
    month_data = monthly_time_spent[monthly_time_spent['Month'] == month]
    axes[idx].set_title(month_names[month - 1])  # Set title to the month name
    custom_pie(axes[idx], month_data, month_data['Exercise'])

axes[num_months].set_title('Yearly Total')
custom_pie(axes[num_months], total_time_spent, total_time_spent['Exercise'])

for j in range(num_months + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
