#%%
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats
import seaborn as sns





#%%
# Read in data, make then EPSG 4326
# From Michael Howe-Ely
df_crime = pd.read_excel('Crime Stats for South Africa.xlsx')
gdf_crime = gpd.GeoDataFrame(df_crime, geometry=gpd.points_from_xy(df_crime.Longitude, df_crime.Latitude)).set_crs(epsg=4326, inplace=True)

# From MNP
# block level
gdf_block_info = gpd.read_parquet("data/MNP/ZAF_geodata.parquet").set_crs(epsg=4326, inplace=True)
# building level
df_bldg_lvl = gpd.read_parquet("data/MNP/buildings_polygons_ZAF.parquet").set_crs(epsg=4326, inplace=True)

# Police station data from SAPS
bounds_shapefile_path = "data/station_boundaries/Police_bounds_.shp"
points_shapefile_path = "data/station_points/Police_points.shp"
bounds_gdf = gpd.read_file(bounds_shapefile_path).set_crs(epsg=4326, inplace=True)
points_gdf = gpd.read_file(points_shapefile_path).set_crs(epsg=4326, inplace=True)
# Crime data from SAPS
# https://www.saps.gov.za/services/crimestats.php, Annual Crime Statistics 2023/2024
crime_excelfile_path = 'data/2023-2024 _Annual_Financial year_WEB1.xlsx'
crime_df = pd.read_excel(crime_excelfile_path, sheet_name=1, skiprows=2, usecols='E:S', nrows=45435)
# 3 stations are non-match: 
# Non-matching values from points_gdf.COMPNT_NM: {'DWARSBERG', 'MAKHAZA', 'SIYATHEMBA'}
# Non-matching values from df.Station: {'DINGLETON'}


 


#%% CLEAN DATA AND JOIN
# Get average total crime for recent 5 years
sum_crime_df = crime_df.groupby(['Station', 'District', 'Province'], as_index=False).sum(numeric_only=True)
sum_crime_df['5yr_avg'] = sum_crime_df[['2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023']].mean(axis=1)

# Join crime stats to police station zones for geometry
bounds_gdf['COMPNT_NM'] = bounds_gdf['COMPNT_NM'].str.upper()
sum_crime_df['Station'] = sum_crime_df['Station'].str.upper()
final_df = pd.merge(sum_crime_df, bounds_gdf[['COMPNT_NM', 'geometry', 'road_length']], left_on='Station', right_on='COMPNT_NM')
final_df['Station'] = final_df['Station'].str.capitalize()
final_df = final_df.drop('COMPNT_NM', axis=1)
final_gdf = gpd.GeoDataFrame(final_df, geometry='geometry').to_crs("EPSG:4326")

# Join blocks to police station zones
intersections = gdf_block_info.sjoin(final_gdf, how="left", predicate="intersects")
projected_crs = "EPSG:3857"
intersections = intersections.to_crs(projected_crs)
intersections['intersection_area'] = intersections.geometry.area
intersections = intersections.reset_index()
intersections = intersections.loc[intersections.groupby('block_id')['intersection_area'].idxmax()].reset_index()
intersections.rename(columns={'Station': 'police_station'}, inplace=True)

# Get building count from building level df
# spatial_join_result = gpd.sjoin(df_bldg_lvl, final_gdf, how='left', op='within')
# bldg_count = spatial_join_result.groupby('index_right').size()
# final_gdf['building_count'] = 0
# final_gdf.loc[building_count.index, 'bldg_count'] = bldg_count.values

# Calculate block agg stats
def weighted_mean(data, weights):
    return (data * weights).sum() / weights.sum()
aggregation_functions = {
    'building_count': 'sum',
    'block_area_km2': 'sum',
    'k_complexity': lambda x: weighted_mean(x, intersections.loc[x.index, 'landscan_population']),
    'landscan_population': 'sum',
    'on_network_street_length_meters': 'sum',
    'off_network_street_length_meters': 'sum'}
aggregated_block_info = intersections.groupby('police_station').agg(aggregation_functions)
final_gdf = final_gdf.merge(aggregated_block_info, left_on='Station', right_index=True, how='left')

# Get average altitude 
import rasterio
import rasterio.mask
from rasterio.mask import mask
from rasterio.plot import show
# Load the DEM data from the .tif file
dem_file = 'data/DEM.tif' # From OpenTopography Copernicus 90M
with rasterio.open(dem_file) as src:
    dem_data = src.read(1)  # Reading the first band
    dem_transform = src.transform
# Define a function to calculate the average altitude within each polygon
def calculate_average_altitude(geom, dem_data, dem_transform):
    with rasterio.open(dem_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True)
        out_image = out_image[0]  #  Taking the first band
        non_masked_data = out_image[out_image != src.nodata]
        if len(non_masked_data) == 0:
            return np.nan  # If there's no valid data, return NaN
        else:
            return non_masked_data.mean()
# Apply the function to each geometry in gdf_units
final_gdf['average_altitude'] = final_gdf['geometry'].apply(
    lambda geom: calculate_average_altitude(geom, dem_data, dem_transform))





#%% DATA TRANSFORMATION + DESCRIPTIVE STATS
# transform data
final_gdf['building_density'] = final_gdf['building_count'] / final_gdf['block_area_km2']
final_gdf['log10_building_density'] = np.log10(final_gdf['building_density'])
final_gdf['total_crime_per_capita'] = final_gdf['5yr_avg'] / final_gdf['landscan_population']
final_gdf['log10_total_crime_per_capita'] = np.log10(final_gdf['total_crime_per_capita'] + 1)
final_gdf['log_landscan_population'] = np.log10(final_gdf['landscan_population'])
final_gdf.describe()

# Plot histograms for each column
columns = ['log10_building_density', 'k_complexity', 'log_landscan_population', 
'on_network_street_length_meters', 'log10_total_crime_per_capita', 'average_altitude']
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))  # 3 rows, 2 columns to accommodate 5 histograms
for i, col in enumerate(columns):
    ax = axes[i//2, i%2]  # Calculate grid position
    ax.hist(final_gdf[col].dropna(), bins=30, edgecolor='k', alpha=0.7)
    ax.set_title(f'Histogram of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
plt.tight_layout()
output_path = 'figs/descriptive_histogram.png'
plt.savefig(output_path, dpi=300)
plt.show()





# %% KNN with individual features
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = ['log10_building_density', 'k_complexity', 'log_landscan_population', 'on_network_street_length_meters', 'average_altitude']
X = final_gdf[features]
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Find the optimal number of clusters using the Elbow method
k_range = range(1, 21)
inertia_values = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Plot for K-Means Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Choose to have 6 clusters
# Plot the box plot of log10_total_crime_per_capita for each cluster
kmeans = KMeans(n_clusters=6, random_state=42)
final_gdf['cluster'] = kmeans.fit_predict(X_scaled)
# Visualize
import matplotlib.patches as mpatches
unique_clusters = sorted(final_gdf['cluster'].unique())
colors = plt.cm.tab20(range(len(unique_clusters)))[:len(unique_clusters)]
cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
# Examine KNN results
# Dependent variable
# sns.boxplot(x ='cluster', hue='cluster', y='log10_total_crime_per_capita', data=final_gdf, palette=list(colors), legend = False)
# plt.title('Box Plot of Crime for Each Cluster, 2019')
# plt.xlabel('Cluster')
# plt.ylabel('log10 (total_crime_per_capita)')
# output_path = 'figs/knn6_cluster_crime_histogram.png'
# plt.savefig(output_path, dpi=300)
# plt.show()

variables = [
    'log10_building_density', 'k_complexity', 
    'log_landscan_population', 'on_network_street_length_meters', 
    'average_altitude', 'log10_total_crime_per_capita']
fig, axes = plt.subplots(2, 3, figsize=(18, 12), tight_layout=True)
for ax, var in zip(axes.flatten(), variables):
    sns.boxplot(x='cluster', y=var, data=final_gdf, ax=ax, palette='Set2')
    ax.set_title(f'Box Plot of {var.replace("_", " ").title()} for Each Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel(var.replace('_', ' ').title())
output_path = 'figs/knn6_cluster_boxplots.png'
plt.savefig(output_path, dpi=300)
plt.show()

# One map
fig, ax = plt.subplots(figsize=(10, 10))
final_gdf['color'] = final_gdf['cluster'].map(cluster_colors)
final_gdf.plot(color=final_gdf['color'], ax=ax)
# Generate custom legend
handles = [mpatches.Patch(color=color, label=f'Cluster {cluster}') for color, cluster in zip(colors[:6], unique_clusters)]
# Title and labels
ax.legend(handles=handles, title="Clusters", loc='lower right')
ax.set_title('Clustered Polygons')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
output_path = 'figs/knn6_result1.png'
plt.savefig(output_path, dpi=300)
plt.show()

# Clusters highlighted individually
# Create a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
axes = axes.flatten()
# Plot each cluster in its own subplot
for i, cluster in enumerate(unique_clusters):
    ax = axes[i]
    final_gdf.plot(color='lightgray', ax=ax)  # Plot all polygons in light gray
    final_gdf[final_gdf['cluster'] == cluster].plot(color=cluster_colors[cluster], ax=ax)  # Highlight the current cluster
    ax.set_title(f'Cluster {cluster}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
output_path = 'figs/knn6_result2.png'
plt.savefig(output_path, dpi=300)
plt.show()

# Summary stats
cluster_summary = final_gdf.groupby('cluster')[features].agg(['mean', 'std'])
print(cluster_summary)

# Check the category (urban / non-urban) of each cluster
intersections = intersections.merge(final_gdf[['cluster', 'Station']], left_on='police_station', right_on='Station', how='left')
summary_table = intersections.groupby(['cluster', 'class_urban_hierarchy']).size().unstack(fill_value=0)





# %%
# LISA
# https://geographicdata.science/book/notebooks/07_local_autocorrelation.html

from pysal.explore import esda
from pysal.lib import weights
w = weights.distance.KNN.from_dataframe(final_gdf, k=6)
w.transform = "R" # Row-standardization
lisa = esda.moran.Moran_Local(final_gdf["log10_total_crime_per_capita"], w)

# plot
from splot import esda as esdaplot
# Set up figure and axes
f, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()

# Subplot 1 #
# Choropleth of local statistics
# Grab first axis in the figure
ax = axs[0]
final_gdf.assign(
    Is=lisa.Is).plot(
    column="Is",
    cmap="plasma",
    scheme="quantiles",
    k=5,
    edgecolor="white",
    linewidth=0.1,
    alpha=0.75,
    legend=True,
    ax=ax,)

# Subplot 2 #
# Quadrant categories
# Grab second axis of local statistics
ax = axs[1]
# Plot Quadrant colors (note to ensure all polygons are assigned a
# quadrant, we "trick" the function by setting significance level to
# 1 so all observations are treated as "significant" and thus assigned
# a quadrant color
esdaplot.lisa_cluster(lisa, final_gdf, p=1, ax=ax)

# Subplot 3 #
# Significance map
# Grab third axis of local statistics
ax = axs[2]
#
# Find out significant observations
labels = pd.Series(
    1 * (lisa.p_sim < 0.05),  # Assign 1 if significant, 0 otherwise
    index=final_gdf.index  # Use the index in the original data
    # Recode 1 to "Significant and 0 to "Non-significant"
).map({1: "Significant", 0: "Non-Significant"})
# Assign labels to `db` on the fly
final_gdf.assign(
    cl=labels
    # Plot choropleth of (non-)significant areas
).plot(
    column="cl",
    categorical=True,
    k=2,
    cmap="Paired",
    linewidth=0.1,
    edgecolor="white",
    legend=True,
    ax=ax,
)


# Subplot 4 #
# Cluster map
# Grab second axis of local statistics
ax = axs[3]
# Plot Quadrant colors In this case, we use a 5% significance
# level to select polygons as part of statistically significant
# clusters
esdaplot.lisa_cluster(lisa, final_gdf, p=0.05, ax=ax)

# Figure styling #
# Set title to each subplot
for i, ax in enumerate(axs.flatten()):
    ax.set_axis_off()
    ax.set_title(
        [
            "Local Statistics",
            "Scatterplot Quadrant",
            "Statistical Significance",
            "Moran Cluster Map",
        ][i],
        y=0,
    )
# Tight layout to minimize in-between white space
f.tight_layout()

# Display the figure
plt.show()



# %%
# test code
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd

def get_road_length_within_polygons(gdf, network_type='all'):
    """
    Calculate the road length within each polygon in the GeoDataFrame.
    
    Parameters:
    gdf (GeoDataFrame): GeoDataFrame containing the polygons.
    network_type (str): The type of street network ('all', 'drive', 'walk', etc.)
    
    Returns:
    GeoDataFrame: The input GeoDataFrame updated with a new column 'road_length' containing road length.
    """
    lengths = []
    
    # https://github.com/JoaoCarabetta/osm-road-length
    
    for polygon in gdf['geometry']:
        # Get the street network within the polygon's bounds
        graph = ox.graph_from_polygon(polygon, network_type=network_type)
        # Get the edges of the graph
        edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        # Clip the edges to the polygon
        clipped_edges = gpd.clip(edges, polygon)
        # Calculate total road length within the polygon
        road_length = clipped_edges.geometry.length.sum()
        lengths.append(road_length)
    # Add the road length to the GeoDataFrame
    gdf['road_length'] = lengths
    
    return gdf

# Example usage
# Calculate road lengths
projected_crs = "EPSG:3857"
final_gdf_ = final_gdf.to_crs(projected_crs)
gdf_with_road_lengths = get_road_length_within_polygons(final_gdf)
# %%
# check diff / make sure the type of road length is correct