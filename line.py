import pandas as pd
import matplotlib.pyplot as plt

def plot_selected_exercises(data, indexes):
    plt.figure(figsize=(10, 6))
    
    for index in indexes:
        if index < len(data):

            angles = [float(angle) for angle in data['Angles'][index].split(',')]
            times = [float(time) for time in data['Times'][index].split(',')]
            
            plt.plot(times, angles, marker='o', linestyle='-', label=f'Exercise {index + 1}')
        else:
            print(f"Index {index} is out of range.")
    
    plt.title('Times vs Angles for Selected Exercises')
    plt.xlabel('Times (seconds)')
    plt.ylabel('Angles (degrees)')
    
    plt.grid()
    
    plt.legend()

    plt.show()

data = pd.read_csv('exercise_data.csv')

selected_indexes = [0, 1]
plot_selected_exercises(data, selected_indexes)