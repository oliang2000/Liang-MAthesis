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
#bounds_shapefile_path = "data/station_boundaries/Police_bounds.shp"
#bounds_gdf = gpd.read_file(bounds_shapefile_path).set_crs(epsg=4326, inplace=True)
bounds_gdf = gpd.read_file("data/station_boundaries/Police_bounds_.gpkg")
points_shapefile_path = "data/station_points/Police_points.shp"
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

# Check the distance distribution
intersections['police_station'] = intersections['police_station'].str.upper()
intersections = intersections.to_crs('EPSG:4326')
points_gdf = points_gdf.to_crs('EPSG:4326')

# Merge intersections with points_gdf to get the location of the assigned police station
merged_df = intersections.merge(points_gdf, left_on='police_station', right_on='COMPNT_NM', suffixes=('_block', '_station'))

# Calculate the distance between each block and its assigned police station in meters
from geopy.distance import geodesic
merged_df['distance_to_station'] = merged_df.apply(
    lambda row: geodesic(
        (row['geometry_block'].centroid.y, row['geometry_block'].centroid.x), 
        (row['geometry_station'].centroid.y, row['geometry_station'].centroid.x)
    ).meters,
    axis=1)
# Plot k_complexity vs. distance to assigned police station
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['distance_to_station'], merged_df['k_complexity'], alpha=0.6)
plt.xlabel('Distance to Assigned Police Station (meters)')
plt.ylabel('K-complexity')
plt.title('K-complexity vs. Distance to Assigned Police Station')
plt.grid(True)
output_path = 'figs/corr_k_dist.png'
plt.savefig(output_path, dpi=300)
plt.show()

# Calculate the correlation between k_complexity and distance_to_station
import statsmodels.api as sm
from scipy import stats
# Dependent variable
y = merged_df['k_complexity']
# Independent variable (with constant term added for the intercept calculation)
X = sm.add_constant(merged_df['distance_to_station'])
# Fit the regression model
model = sm.OLS(y, X).fit()
# Calculate the R-squared
r_squared = model.rsquared
# Calculate the Pearson correlation coefficient
correlation = merged_df['k_complexity'].corr(merged_df['distance_to_station'])
# Obtain the slope (coefficient of distance_to_station) and its 95% CI
slope = model.params['distance_to_station']
conf_int = model.conf_int().loc['distance_to_station']
# Format the output
print(f"R^2: {r_squared}")
print(f"Correlation: {correlation}")
print(f"Slope: {slope}")
print(f"95% CI for slope: {conf_int[0]} to {conf_int[1]}")


# Plot boxplots of k_complexity for each distance bin
plt.figure(figsize=(10, 6))
merged_df.boxplot(column='k_complexity', by='distance_bin', grid=False)
plt.xlabel('Distance Bin')
plt.ylabel('K-complexity')
plt.title('K-complexity by Distance Bin')
plt.suptitle('')  # Suppress default titles
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


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
final_gdf['log10_landscan_population_density'] = np.log10(final_gdf['landscan_population'] / final_gdf['block_area_km2'])
final_gdf.describe()

# Plot histograms for each column
variables = ['log10_building_density', 'k_complexity', 'log10_landscan_population_density', 
'road_length', 'average_altitude', 'log10_total_crime_per_capita']
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))  # 3 rows, 2 columns to accommodate 5 histograms
for i, col in enumerate(variables):
    ax = axes[i//2, i%2]  # Calculate grid position
    ax.hist(final_gdf[col].dropna(), bins=30, edgecolor='k', alpha=0.7)
    ax.set_title(f'Histogram of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
plt.tight_layout()
output_path = 'figs/descriptive_histogram.png'
plt.savefig(output_path, dpi=300)
plt.show()

# Plot map for each column
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15)) 
for i, col in enumerate(variables):
    ax = axes[i//2, i%2]  # Calculate grid position
    final_gdf.plot(column=col, ax=ax, legend=True, cmap='viridis', 
                   legend_kwds={'shrink': 0.5}, 
                   missing_kwds={"color": "lightgrey", "edgecolor": "red", "hatch": "///"})
    ax.set_title(f'Map of {col}')
    ax.axis('off')
plt.tight_layout()
output_path = 'figs/descriptive_map.png'
plt.savefig(output_path, dpi=300)
plt.show()




# %% KNN with individual features
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = ['log10_building_density', 'k_complexity', 'log10_landscan_population_density', 'road_length', 'average_altitude']
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
output_path = 'figs/elbow_plot.png'
plt.savefig(output_path, dpi=300)
plt.grid(True)
plt.show()


# Choose 7 clusters
kmeans = KMeans(n_clusters=7, random_state=42)
final_gdf['cluster_knn7'] = kmeans.fit_predict(X_scaled)

# VISUALIZE
# Color palette
import matplotlib.patches as mpatches
unique_clusters = sorted(final_gdf['cluster_knn7'].unique())
colors = plt.cm.tab20(range(len(unique_clusters)))[:len(unique_clusters)]
cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}

# Plot the box plot of all 5 dependent variables, 1 independent variable for all clusters
palette = {str(cluster): color for cluster, color in cluster_colors.items()}
fig, axes = plt.subplots(2, 3, figsize=(18, 12), tight_layout=True)
# Plot each variable in a separate subplot
for ax, var in zip(axes.flatten(), variables):
    sns.boxplot(x='cluster_knn7', y=var, data=final_gdf, ax=ax, palette=palette)
    ax.set_title(f'Box Plot of {var.replace("_", " ").title()} for Each Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel(var.replace('_', ' ').title())
# Add a custom legend for clusters
handles = [mpatches.Patch(color=color, label=f'Cluster {cluster}') for cluster, color in cluster_colors.items()]
# Use `bbox_transform` to position the legend outside the axes on the right side
fig.legend(handles=handles, title='Clusters', loc='center left', bbox_to_anchor=(1.02, 0.5))
output_path = 'figs/knn7_cluster_boxplots.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

# Bar plot for counts
df_count = final_gdf.groupby('cluster_knn7').size().reset_index().rename(columns={0: 'count'})
fig, ax = plt.subplots()
sns.barplot(x='cluster_knn7', y="count", data=df_count, ax=ax, palette=palette)
ax.set_xlabel('Cluster')
ax.set_ylabel('Count')

# One map
fig, ax = plt.subplots(figsize=(10, 10))
final_gdf['color'] = final_gdf['cluster_knn7'].map(cluster_colors)
final_gdf.plot(color=final_gdf['color'], ax=ax)
# Generate custom legend
handles = [mpatches.Patch(color=color, label=f'Cluster {cluster}') for color, cluster in zip(colors, unique_clusters)]
# Title and labels
ax.legend(handles=handles, title="Clusters", loc='lower right')
ax.set_title('KNN Clustering of South African Police Station Areas')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
output_path = 'figs/knn7_map.png'
plt.savefig(output_path, dpi=300)
plt.show()

# Clusters highlighted individually
# fig, axes = plt.subplots(2, 4, figsize=(18, 12), sharex=True, sharey=True)
# axes = axes.flatten()
# # Plot each cluster in its own subplot
# for i, cluster in enumerate(unique_clusters):
#     ax = axes[i]
#     final_gdf.plot(color='lightgray', ax=ax)  # Plot all polygons in light gray
#     final_gdf[final_gdf['cluster_knn5'] == cluster].plot(color=cluster_colors[cluster], ax=ax)  # Highlight the current cluster
#     ax.set_title(f'Cluster {cluster}')
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
# # Hide any unused subplots
# for j in range(i + 1, len(axes)):
#     axes[j].axis('off')
# plt.tight_layout()
# output_path = 'figs/knn7_result2.png'
# plt.savefig(output_path, dpi=300)
# plt.show()

# Summary stats
cluster_summary = final_gdf.groupby('cluster_knn7')[features].agg(['mean', 'std'])
print(cluster_summary)

# Check the category (urban / non-urban) of each cluster
intersections = intersections.merge(final_gdf[['cluster_knn5', 'Station']], left_on='police_station', right_on='Station', how='left')
summary_table = intersections.groupby(['cluster_knn5', 'class_urban_hierarchy']).size().unstack(fill_value=0)





# %%
# LISA: local moran's i
# https://geographicdata.science/book/notebooks/07_local_autocorrelation.html

from pysal.explore import esda
from pysal.lib import weights
w = weights.distance.KNN.from_dataframe(final_gdf, k=7)
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
final_gdf.assign(Is=lisa.Is).plot(column="Is", cmap="plasma", scheme="quantiles", k=5, edgecolor="white", linewidth=0.1, alpha=0.75, legend=True, ax=ax)
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
# Find out significant observations
labels = pd.Series(
    1 * (lisa.p_sim < 0.05),  # Assign 1 if significant, 0 otherwise
    index=final_gdf.index  # Use the index in the original data
    # Recode 1 to "Significant and 0 to "Non-significant"
).map({1: "Significant", 0: "Non-Significant"})
# Assign labels to `db` on the fly
final_gdf.assign(cl=labels).plot(column="cl",categorical=True,k=2,cmap="Paired",linewidth=0.1,edgecolor="white",legend=True,ax=ax,)
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
f.tight_layout()
plt.show()























