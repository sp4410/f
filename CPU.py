import pandas as pd
import folium
from folium.plugins import HeatMap

# Path to the data file
data_file = r"C:\Users\ThinkPad\Desktop\f\Motor_Vehicle_Collisions_-_Crashes_20241107.csv"

# Load the dataset, set low_memory=False to avoid warnings
data = pd.read_csv(data_file, low_memory=False)

# Clean the data: Remove invalid latitude and longitude entries
data_geo = data.dropna(subset=['LATITUDE', 'LONGITUDE'])
data_geo = data_geo[(data_geo['LATITUDE'] != 0.0) & (data_geo['LONGITUDE'] != 0.0)]

# Generate Heatmap HTML
print("Generating Heatmap HTML...")
m_heatmap = folium.Map(location=[40.730610, -73.935242], zoom_start=10)  # Centered at NYC

# Create Heatmap data
heat_data = [[row['LATITUDE'], row['LONGITUDE']] for index, row in data_geo.iterrows()]
HeatMap(heat_data, radius=8, max_zoom=13).add_to(m_heatmap)

# Save Heatmap HTML
heatmap_file = r"C:\Users\ThinkPad\Desktop\f\model&img\Heatmap_cpu.html"
m_heatmap.save(heatmap_file)
print(f"Heatmap has been generated and saved to: {heatmap_file}")

# Generate Markers Map HTML
print("Generating Markers Map HTML...")
m_markers = folium.Map(location=[40.730610, -73.935242], zoom_start=10)  # Centered at NYC

# Sample data for visualization performance optimization
sample_data_severity = data_geo.sample(n=1000, random_state=42)

# Add red, orange, and green markers
for index, row in sample_data_severity.iterrows():
    try:
        if row['NUMBER OF PERSONS KILLED'] > 0:
            # Red triangle (fatal accidents)
            folium.features.RegularPolygonMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                number_of_sides=3,  # Triangle
                radius=13,
                color="red",
                fill=True,
                fill_color="red"
            ).add_to(m_markers)
        elif row['NUMBER OF PERSONS INJURED'] > 0:
            # Orange circle (injury accidents)
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                radius=5,
                color="orange",
                fill=True,
                fill_color="orange"
            ).add_to(m_markers)
        else:
            # Green square (no injury or fatality)
            folium.features.RegularPolygonMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                number_of_sides=4,  # Square
                radius=5,
                color="green",
                fill=True,
                fill_color="green"
            ).add_to(m_markers)
    except KeyError as e:
        print(f"Missing key column: {e}")
    except ValueError as e:
        print(f"Data error: {e}")

# Save Markers Map HTML
markers_file = r"C:\Users\ThinkPad\Desktop\f\model&img\Markers_cpu.html"
m_markers.save(markers_file)
print(f"Markers map has been generated and saved to: {markers_file}")
