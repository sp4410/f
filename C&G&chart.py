import pandas as pd
import torch
import time
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Path to the data file
data_file = r"C:\Users\ThinkPad\Desktop\f\Motor_Vehicle_Collisions_-_Crashes_20241107.csv"

# Load the dataset
data = pd.read_csv(data_file, low_memory=False)

# Clean the data: Remove invalid latitude and longitude entries
data_geo = data.dropna(subset=['LATITUDE', 'LONGITUDE'])
data_geo = data_geo[(data_geo['LATITUDE'] != 0.0) & (data_geo['LONGITUDE'] != 0.0)]

# =========== GPU Acceleration =============
print("Starting GPU processing...")
start_gpu = time.time()

# Convert to PyTorch tensors and perform marker classification on GPU
latitudes = torch.tensor(data_geo['LATITUDE'].values, device='cuda')
longitudes = torch.tensor(data_geo['LONGITUDE'].values, device='cuda')
killed = torch.tensor(data_geo['NUMBER OF PERSONS KILLED'].values, device='cuda')
injured = torch.tensor(data_geo['NUMBER OF PERSONS INJURED'].values, device='cuda')

# Classify markers
markers_gpu = torch.empty(len(data_geo), dtype=torch.int32, device='cuda')  # 0: green, 1: orange, 2: red
markers_gpu[killed > 0] = 2  # Red triangle (fatal accidents)
markers_gpu[(killed == 0) & (injured > 0)] = 1  # Orange circle (injury accidents)
markers_gpu[(killed == 0) & (injured == 0)] = 0  # Green square (no injuries or fatalities)

# Download results back to CPU
markers_gpu_cpu = markers_gpu.cpu().numpy()

gpu_time = time.time() - start_gpu
print(f"GPU processing time: {gpu_time:.4f} seconds")

# =========== CPU Iteration =============
print("Starting CPU processing...")
start_cpu = time.time()

# Use DataFrame iteration to classify markers row by row
marker_types_cpu = []
for index, row in data_geo.iterrows():
    if row['NUMBER OF PERSONS KILLED'] > 0:
        marker_types_cpu.append(2)  # Red triangle
    elif row['NUMBER OF PERSONS INJURED'] > 0:
        marker_types_cpu.append(1)  # Orange circle
    else:
        marker_types_cpu.append(0)  # Green square

cpu_time = time.time() - start_cpu
print(f"CPU processing time: {cpu_time:.4f} seconds")

# =========== Performance and Metrics Evaluation =============
print("Calculating performance metrics...")

# Accuracy
accuracy = accuracy_score(marker_types_cpu, markers_gpu_cpu)
print(f"Accuracy: {accuracy:.4f}")

# Recall
recall = recall_score(marker_types_cpu, markers_gpu_cpu, average='macro')
print(f"Recall: {recall:.4f}")

# F1 Score
f1 = f1_score(marker_types_cpu, markers_gpu_cpu, average='macro')
print(f"F1 Score: {f1:.4f}")

# Plot comparison charts
print("Plotting comparison charts...")

# Simulated loss reduction data (time as loss value)
methods = ['GPU', 'CPU']
times = [gpu_time, cpu_time]

# Plot accuracy, recall, and F1-score comparison
metrics = ['Accuracy', 'Recall', 'F1-Score']
metric_values = [accuracy, recall, f1]

plt.figure(figsize=(10, 5))

# Time comparison chart
plt.subplot(1, 2, 1)
plt.bar(methods, times, color=['blue', 'orange'])
plt.ylabel("Time (seconds)")
plt.title("GPU vs CPU Time Comparison")

# Metrics comparison chart
plt.subplot(1, 2, 2)
plt.plot(metrics, metric_values, marker='o', color='green', label='Metrics')
plt.ylim(0, 1.1)
plt.ylabel("Value")
plt.title("GPU Metrics Comparison")
plt.legend()

# Show charts
plt.tight_layout()
plt.show()

# Performance comparison
print(f"GPU is {cpu_time / gpu_time:.2f} times faster than CPU")
