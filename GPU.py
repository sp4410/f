import pandas as pd
import folium
import torch

# Path to the data file
data_file = r"C:\Users\ThinkPad\Desktop\f\Motor_Vehicle_Collisions_-_Crashes_20241107.csv"

# Load the dataset
data = pd.read_csv(data_file, low_memory=False)

# Clean the data: Remove invalid latitude and longitude entries
data_geo = data.dropna(subset=['LATITUDE', 'LONGITUDE'])
data_geo = data_geo[(data_geo['LATITUDE'] != 0.0) & (data_geo['LONGITUDE'] != 0.0)]

# Limit the data size (sampling)
sample_size = 2000
data_geo_sampled = data_geo.sample(n=sample_size, random_state=42)

# Convert to PyTorch tensors
latitudes = torch.tensor(data_geo_sampled['LATITUDE'].values, device='cuda')
longitudes = torch.tensor(data_geo_sampled['LONGITUDE'].values, device='cuda')
killed = torch.tensor(data_geo_sampled['NUMBER OF PERSONS KILLED'].values, device='cuda')
injured = torch.tensor(data_geo_sampled['NUMBER OF PERSONS INJURED'].values, device='cuda')

# Use GPU to compute marker categories
print("Using GPU acceleration to generate marker types...")
markers = torch.empty(len(data_geo_sampled), dtype=torch.int32, device='cuda')
markers[killed > 0] = 2  # Red triangle
markers[(killed == 0) & (injured > 0)] = 1  # Orange circle
markers[(killed == 0) & (injured == 0)] = 0  # Green square

# Transfer results back to CPU
data_geo_sampled['MARKER_TYPE'] = markers.cpu().numpy()

# Create a marker map (GPU)
m_markers_gpu = folium.Map(location=[40.730610, -73.935242], zoom_start=10)

# Define marker styles
marker_styles = {
    2: {"shape": "triangle", "color": "red", "size": 13},  # Red triangle
    1: {"shape": "circle", "color": "orange", "size": 5},  # Orange circle
    0: {"shape": "square", "color": "green", "size": 5},  # Green square
}

# Iterate over the data and add markers
for index, row in data_geo_sampled.iterrows():
    location = [row['LATITUDE'], row['LONGITUDE']]
    marker_type = row['MARKER_TYPE']
    style = marker_styles[marker_type]

    if style["shape"] == "triangle":
        folium.features.RegularPolygonMarker(
            location=location,
            number_of_sides=3,
            radius=style["size"],
            color=style["color"],
            fill=True,
            fill_color=style["color"]
        ).add_to(m_markers_gpu)
    elif style["shape"] == "circle":
        folium.CircleMarker(
            location=location,
            radius=style["size"],
            color=style["color"],
            fill=True,
            fill_color=style["color"]
        ).add_to(m_markers_gpu)
    elif style["shape"] == "square":
        folium.features.RegularPolygonMarker(
            location=location,
            number_of_sides=4,
            radius=style["size"],
            color=style["color"],
            fill=True,
            fill_color=style["color"]
        ).add_to(m_markers_gpu)

# Save the GPU marker map
gpu_file = r"C:\Users\ThinkPad\Desktop\f\model&img\Markers_gpu.html"
m_markers_gpu.save(gpu_file)
print(f"GPU marker map has been generated and saved to: {gpu_file}")
