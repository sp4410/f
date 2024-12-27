import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = r"C:\Users\ThinkPad\Desktop\mid2\1107.csv"
data = pd.read_csv(file_path)

# Convert 'CRASH DATE' and 'CRASH TIME' columns to datetime format
data['CRASH DATE'] = pd.to_datetime(data['CRASH DATE'])
data['CRASH TIME'] = pd.to_datetime(data['CRASH TIME'], format='%H:%M')

# Extract hour information
data['Hour of Day'] = data['CRASH TIME'].dt.hour

# Shift the day's starting point to 5 AM
data['Shifted Hour'] = (data['Hour of Day'] - 5) % 24

# Group by 'Shifted Hour' and calculate the number of crashes per hour
shifted_crashes_per_hour = data.groupby('Shifted Hour').size()
hours = shifted_crashes_per_hour.index
crashes = shifted_crashes_per_hour.values

# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Initial guess for the parameters
initial_guess = [max(crashes), 16, 4]

# Fit the Gaussian function using least squares
params, _ = curve_fit(gaussian, hours, crashes, p0=initial_guess)
amplitude, mean, stddev = params

# Generate the fitted curve
x = np.linspace(min(hours), max(hours), 100)
fitted_curve = gaussian(x, amplitude, mean, stddev)

# Plot the fitted Gaussian distribution
plt.figure(figsize=(12, 6))
plt.plot(x, fitted_curve, color='red', linewidth=2, label='Fitted Gaussian')
plt.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.2f}')
plt.text(mean + 1, max(fitted_curve) * 0.8, f'Std: {stddev:.2f}', fontsize=12, color='black')

# Display the Gaussian expression
expression = f'Gaussian: {amplitude:.2f} * exp(-((x - {mean:.2f})^2) / (2 * {stddev:.2f}^2))'
plt.text(min(x), max(fitted_curve) * 0.6, expression, fontsize=12, color='green')

# Add title and labels
plt.title('Fitted Normal Distribution with Expression', fontsize=16)
plt.xlabel('Shifted Hour of Day (5AM to 4AM)', fontsize=14)
plt.ylabel('Fitted Values', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
