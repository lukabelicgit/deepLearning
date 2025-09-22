import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("exercise_data.csv")

df['Date'] = pd.to_datetime(df['Date'])

exercises = df['Exercise'].unique()

for exercise in exercises:
    exercise_data = df[df['Exercise'].str.lower() == exercise.lower()]
    
    daily_reps = exercise_data.groupby("Date")["Reps"].sum().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.bar(daily_reps['Date'], daily_reps['Reps'], color='skyblue', width=1.6)
    
    plt.title(f'Total Repetitions for {exercise.capitalize()} by Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Repetitions')
    plt.xticks(rotation=90)

    # Show the plot
    plt.tight_layout()
    plt.show()
