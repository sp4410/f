import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = r"C:\Users\ThinkPad\Desktop\f\Motor_Vehicle_Collisions_-_Crashes_20241107.csv"
data = pd.read_csv(file_path)

# Convert 'CRASH DATE' and 'CRASH TIME' columns to datetime format
data['CRASH DATE'] = pd.to_datetime(data['CRASH DATE'])
data['CRASH TIME'] = pd.to_datetime(data['CRASH TIME'], format='%H:%M')

# Extract hour information
data['Hour of Day'] = data['CRASH TIME'].dt.hour

# Shift the day's starting point to 5 AM
data['Shifted Hour'] = (data['Hour of Day'] - 5) % 24

# Group by 'Shifted Hour' and count the number of crashes per hour
shifted_crashes_per_hour = data.groupby('Shifted Hour').size()

# Plot the bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x=shifted_crashes_per_hour.index, 
            y=shifted_crashes_per_hour.values, 
            palette="viridis")

# Add title and axis labels
plt.title('Number of Crashes per Hour (5AM to 4AM)', fontsize=16)
plt.xlabel('Shifted Hour of Day (5AM to 4AM)', fontsize=14)
plt.ylabel('Number of Crashes', fontsize=14)

# Adjust x-axis tick labels to display original hours (starting from 5 AM)
plt.xticks(range(len(shifted_crashes_per_hour.index)), 
           labels=[(hour + 5) % 24 for hour in shifted_crashes_per_hour.index])

plt.tight_layout()
plt.show()
