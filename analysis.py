#%%
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats
import seaborn as sns



#%% Read in data, make then EPSG 4326

#############################
##### DATA PROCESSING #######
#############################

# From MNP
# block level
gdf_block_info = gpd.read_parquet("data/MNP/ZAF_geodata.parquet").set_crs(epsg=4326, inplace=True)
# building level
df_bldg_lvl = gpd.read_parquet("data/MNP/buildings_polygons_ZAF.parquet").set_crs(epsg=4326, inplace=True)

# From SAPS
# Police station data 
bounds_gdf = gpd.read_file("data/station_boundaries/Police_bounds_.gpkg")
points_shapefile_path = "data/station_points/Police_points.shp"
points_gdf = gpd.read_file(points_shapefile_path).set_crs(epsg=4326, inplace=True)
# Crime data
#https://www.saps.gov.za/services/older_crimestats.php, Annual Crime Statistics 2023/2024
crime_excelfile_path = 'data/2023-2024 _Annual_Financial year_WEB1.xlsx'
crime_df = pd.read_excel(crime_excelfile_path, sheet_name=1, skiprows=2, usecols='E:S', nrows=45435)
# 3 stations are non-match: 
# Non-matching values from points_gdf.COMPNT_NM: {'DWARSBERG', 'MAKHAZA', 'SIYATHEMBA'}
# Non-matching values from df.Station: {'DINGLETON'}



#%% CLEAN DATA AND JOIN
########################
# Get average total crime for recent 5 years
sum_crime_df = crime_df.groupby(['Station', 'District', 'Province'], as_index=False).sum(numeric_only=True)
sum_crime_df['5yr_avg'] = sum_crime_df[['2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023']].mean(axis=1) # april 2022 to march 2023

# Join crime stats to police station zones for geometry
bounds_gdf['COMPNT_NM'] = bounds_gdf['COMPNT_NM'].str.upper()
sum_crime_df['Station'] = sum_crime_df['Station'].str.upper()
final_df = pd.merge(sum_crime_df, bounds_gdf[['COMPNT_NM', 'geometry', 'road_length']], left_on='Station', right_on='COMPNT_NM')
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
intersections['police_station'] = intersections['police_station'].str.upper()
intersections = intersections.to_crs('EPSG:4326')
points_gdf = points_gdf.to_crs('EPSG:4326')
# Merge intersections with points_gdf to get the location of the assigned police station
merged_df = intersections.merge(points_gdf, left_on='police_station', right_on='COMPNT_NM', suffixes=('_block', '_station'))


################################
# Check the distance distribution
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
plt.xlabel('Distance to Assigned Police Station (meters)', fontsize=14)
plt.ylabel('K-complexity', fontsize=14)
plt.title('K-complexity vs. Distance to Assigned Police Station', fontsize=16, pad=15)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
output_path = 'figs/corr_k_dist.png'
plt.savefig(output_path, dpi=300)
plt.show()


# Calculate the correlation between k_complexity and distance_to_station
y = merged_df['k_complexity']
X = sm.add_constant(merged_df['distance_to_station'])
model = sm.OLS(y, X).fit()
r_squared = model.rsquared
correlation = merged_df['k_complexity'].corr(merged_df['distance_to_station'])
slope = model.params['distance_to_station']
conf_int = model.conf_int().loc['distance_to_station']
# Format the output
print(f"R^2: {r_squared}")
print(f"Correlation: {correlation}")
print(f"Slope: {slope}")
print(f"95% CI for slope: {conf_int[0]} to {conf_int[1]}")



# %% CAUSAL ANALYSIS

#######################
### CAUSAL ANALYSIS ###
#######################

from statsmodels.tsa.stattools import grangercausalitytests

# Data
# Get building counts
western_cape_gdf = final_gdf[final_gdf["Province"] == "Western Cape"]
# Save the GeoDataFrame as a shapefile
#output_file_path = "data/police_areas_westerncape.shp"
#western_cape_gdf.to_file(output_file_path, driver="ESRI Shapefile")
western_cape_building_growth = pd.read_csv("data/westerncape_bldg_footprint.csv")
# clean tables
# crime table: april 2022 to march 2023 is "2022-2023", rename this to crime_2022
western_cape_gdf = western_cape_gdf.rename(columns=lambda x: f"crime_{x.split('-')[0]}" if '-' in x else x)
# building count table: calculate growth rates by year
for year in range(2016, 2023):
    western_cape_building_growth[f'growth_{year+1}'] = (western_cape_building_growth[f'count_{year + 1}'] / western_cape_building_growth[f'count_{year}']) - 1
# merge
# contains: crime 2014-2023, growth 2017 to 2023
merged_data = western_cape_gdf.merge(western_cape_building_growth, on='Station')

# calculate causality
granger_results_crime_to_growth = []
granger_results_growth_to_crime = []
for station in merged_data['Station'].unique():
    for i in range(2):
        station_data = merged_data[merged_data['Station'] == station]
        station_data = station_data[[col for col in station_data.columns if col.startswith('crime_') or col.startswith('growth_')]]
        melted_data = pd.melt(station_data, var_name='Year', value_name='Value')
        melted_data[['Type', 'Year']] = melted_data['Year'].str.split('_', expand=True)
        pivoted_data = melted_data.pivot(index='Year', columns='Type', values='Value').reset_index().fillna(0)
        pivoted_data['Year'] = pivoted_data['Year'].astype(int)
        if i:
            # Perform Granger Causality test for crime -> growth for lags 1, 2, and 3
            lag_range = 2
            test_result = grangercausalitytests(pivoted_data[pivoted_data["Year"]>=2016][["crime", "growth"]], lag_range, verbose = False)
            for lag in range(lag_range):
                lag = lag + 1
                granger_results_crime_to_growth.append([station, test_result[lag][0]['ssr_ftest'][0], test_result[lag][0]['ssr_ftest'][1], lag])
        else:
            # Perform Granger Causality test for growth -> crime for lags 1, 2, and 3
            lag = 1
            test_result = grangercausalitytests(pivoted_data[pivoted_data["Year"]>=2017][["growth", "crime"]], lag, verbose = False)
            granger_results_growth_to_crime.append([station, test_result[lag][0]['ssr_ftest'][0], test_result[lag][0]['ssr_ftest'][1], lag])
df_granger_crime_to_growth = pd.DataFrame(granger_results_crime_to_growth, columns=['Station', 'f_statistic', 'p_value', 'lag'])
df_granger_growth_to_crime = pd.DataFrame(granger_results_growth_to_crime, columns=['Station', 'f_statistic', 'p_value', 'lag'])

# Plot histograms of p_value for each lag
for lag in df_granger_crime_to_growth['lag'].unique():
    subset = df_granger_crime_to_growth[df_granger_crime_to_growth['lag'] == lag]
    plt.figure(figsize=(8, 4))
    plt.hist(subset['p_value'], bins=50, edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of p_value for lag {lag}, Crime -> Growth')
    plt.xlabel('p_value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# Plot histograms of p_value for each lag
for lag in df_granger_growth_to_crime['lag'].unique():
    subset = df_granger_growth_to_crime[df_granger_growth_to_crime['lag'] == lag]
    plt.figure(figsize=(8, 4))
    plt.hist(subset['p_value'], bins=50, edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of p_value for lag {lag}, Growth -> Crime')
    plt.xlabel('p_value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

df_granger_crime_to_growth[(df_granger_crime_to_growth['lag'] == 1) & (df_granger_crime_to_growth['p_value'] < 0.05)]
df_granger_growth_to_crime[(df_granger_growth_to_crime['lag'] == 1) & (df_granger_growth_to_crime['p_value'] < 0.05)]





# THE SAME ANALYSIS FOR KwaZulu-Natal
# Get building counts
KwaZulu_Natal_gdf = final_gdf[final_gdf["Province"] == "KwaZulu-Natal"]
# Save the GeoDataFrame as a shapefile
# KwaZulu_Natal_gdf.to_file("data/police_areas_KN.shp", driver="ESRI Shapefile")
KN_building_growth = pd.read_csv("data/KN_bldg_footprint.csv")
# clean tables
# crime table: april 2022 to march 2023 is "2022-2023", rename this to crime_2022
KwaZulu_Natal_gdf = KwaZulu_Natal_gdf.rename(columns=lambda x: f"crime_{x.split('-')[0]}" if '-' in x else x)
# building count table: calculate growth rates by year
for year in range(2016, 2023):
    KN_building_growth[f'growth_{year+1}'] = (KN_building_growth[f'count_{year + 1}'] / KN_building_growth[f'count_{year}']) - 1
# merge
# contains: crime 2014-2023, growth 2017 to 2023
merged_data = KwaZulu_Natal_gdf.merge(KN_building_growth, on='Station')

# calculate causality
granger_results_crime_to_growth = []
granger_results_growth_to_crime = []
for station in merged_data['Station'].unique():
    for i in range(2):
        station_data = merged_data[merged_data['Station'] == station]
        station_data = station_data[[col for col in station_data.columns if col.startswith('crime_') or col.startswith('growth_')]]
        melted_data = pd.melt(station_data, var_name='Year', value_name='Value')
        melted_data[['Type', 'Year']] = melted_data['Year'].str.split('_', expand=True)
        pivoted_data = melted_data.pivot(index='Year', columns='Type', values='Value').reset_index().fillna(0)
        pivoted_data['Year'] = pivoted_data['Year'].astype(int)
        if i:
            # Perform Granger Causality test for crime -> growth for lags 1, 2, and 3
            lag_range = 2
            test_result = grangercausalitytests(pivoted_data[pivoted_data["Year"]>=2016][["crime", "growth"]], lag_range, verbose = False)
            for lag in range(lag_range):
                lag = lag + 1
                granger_results_crime_to_growth.append([station, test_result[lag][0]['ssr_ftest'][0], test_result[lag][0]['ssr_ftest'][1], lag])
        else:
            # Perform Granger Causality test for growth -> crime for lags 1, 2, and 3
            lag = 1
            test_result = grangercausalitytests(pivoted_data[pivoted_data["Year"]>=2017][["growth", "crime"]], lag, verbose = False)
            granger_results_growth_to_crime.append([station, test_result[lag][0]['ssr_ftest'][0], test_result[lag][0]['ssr_ftest'][1], lag])
df_granger_crime_to_growth = pd.DataFrame(granger_results_crime_to_growth, columns=['Station', 'f_statistic', 'p_value', 'lag'])
df_granger_growth_to_crime = pd.DataFrame(granger_results_growth_to_crime, columns=['Station', 'f_statistic', 'p_value', 'lag'])

# Plot histograms of p_value for each lag
for lag in df_granger_crime_to_growth['lag'].unique():
    subset = df_granger_crime_to_growth[df_granger_crime_to_growth['lag'] == lag]
    plt.figure(figsize=(8, 4))
    plt.hist(subset['p_value'], bins=50, edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of p_value for lag {lag}, Crime -> Growth')
    plt.xlabel('p_value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# Plot histograms of p_value for each lag
for lag in df_granger_growth_to_crime['lag'].unique():
    subset = df_granger_growth_to_crime[df_granger_growth_to_crime['lag'] == lag]
    plt.figure(figsize=(8, 4))
    plt.hist(subset['p_value'], bins=50, edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of p_value for lag {lag}, Growth -> Crime')
    plt.xlabel('p_value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

df_granger_crime_to_growth[(df_granger_crime_to_growth['lag'] == 1) & (df_granger_crime_to_growth['p_value'] < 0.05)] #4
df_granger_growth_to_crime[(df_granger_growth_to_crime['lag'] == 1) & (df_granger_growth_to_crime['p_value'] < 0.05)] #8


# Update station names to camel case
station_names = {
    "MID ILLOVO": "Mid Illovo",
    "KWAMBONAMBI": "Kwambonambi"
}
# Set font size for all text
plt.rcParams.update({'font.size': 12})
for station in ["MID ILLOVO", "KWAMBONAMBI"]:
    station_data = merged_data[merged_data['Station'] == station]
    station_data = station_data[[col for col in station_data.columns if col.startswith('crime_') or col.startswith('growth_')]]
    melted_data = pd.melt(station_data, var_name='Year', value_name='Value')
    melted_data[['Type', 'Year']] = melted_data['Year'].str.split('_', expand=True)
    pivoted_data = melted_data.pivot(index='Year', columns='Type', values='Value').reset_index().fillna(0)
    pivoted_data['Year'] = pivoted_data['Year'].astype(int)
    pivoted_data = pivoted_data[pivoted_data["Year"] > 2016]
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(pivoted_data['Year'], pivoted_data['crime'], marker='o', color='blue')
    ax1.set_title('Crime Over Years', fontsize=16)
    ax1.set_ylabel('Crime', fontsize=14)
    ax1.grid(True)
    ax2.plot(pivoted_data['Year'], pivoted_data['growth'], marker='x', color='green')
    ax2.set_title('Growth Over Years', fontsize=16)
    ax2.set_xlabel('Year', fontsize=14)
    ax2.set_ylabel('Growth', fontsize=14)
    ax2.grid(True)
    fig.suptitle(f'Crime and Growth Trend for {station_names[station]}', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'figs/crime_growth_{station_names[station]}.png')
    plt.show()


# overall distribution of growth
years = [str(year) for year in range(2017, 2024)]
growth_columns = [f'growth_{year}' for year in years]
growth_data = merged_data[growth_columns]
# Set the figure size
plt.figure(figsize=(12, 8))
growth_data.plot(kind='box', grid=True, ax=plt.gca())
plt.title('Growth Distribution from 2017 to 2023')
plt.xlabel('Year')
plt.ylabel('Growth')
plt.show()

growth_data = merged_data[['Station'] + growth_columns]
# Set the figure size
plt.figure(figsize=(12, 8))
for station in growth_data['Station'].unique():
    station_data = growth_data[growth_data['Station'] == station]
    plt.plot(years, station_data[growth_columns].values.flatten(), marker='o', color='blue', alpha=0.5)
plt.title('Growth from 2017 to 2023 for Each Station')
plt.xlabel('Year')
plt.ylabel('Growth')
plt.grid(True)
plt.show()







#%%

#########################################
#DATA TRANSFORMATION + DESCRIPTIVE STATS
#########################################

import rasterio
import rasterio.mask
from rasterio.mask import mask
from rasterio.plot import show

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
aggregated_block_info = intersections.groupby('police_station').agg(aggregation_functions).reset_index()
final_gdf = final_gdf.merge(aggregated_block_info, left_on='Station', right_on = "police_station", how='left') #right_index=True

# Get average altitude 
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


# transform data
final_gdf['building_density'] = final_gdf['building_count'] / final_gdf['block_area_km2']
final_gdf['log10_building_density'] = np.log10(final_gdf['building_density'])
final_gdf['total_crime_per_capita'] = final_gdf['5yr_avg'] / final_gdf['landscan_population']
final_gdf['log10_total_crime_per_capita'] = np.log10(final_gdf['total_crime_per_capita'] + 1)
final_gdf['log10_landscan_population_density'] = np.log10(final_gdf['landscan_population'] / final_gdf['block_area_km2'])
final_gdf.describe()


#%% 

###################################
##############  EDA  ##############
###################################

# Plot histograms for each column
variables = ['log10_building_density', 'k_complexity', 'log10_landscan_population_density', 
             'road_length', 'average_altitude', 'log10_total_crime_per_capita']
feature_labels = {
    'log10_building_density': 'Log10 Building\nDensity',
    'k_complexity': 'K-Complexity',
    'log10_landscan_population_density': 'Log10 LandScan Population Density',
    'road_length': 'Road Length',
    'average_altitude': 'Average Altitude',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'log10_total_crime_per_capita': 'Log10 Total Crime Per Capita'
}
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))  # 2 rows, 3 columns to accommodate 6 histograms
for i, col in enumerate(variables):
    ax = axes[i//3, i%3]  # Calculate grid position
    ax.hist(final_gdf[col].dropna(), bins=30, edgecolor='k', alpha=0.7)
    ax.set_xlabel(feature_labels[col], fontsize=16)  # Use feature labels for x-axis labels
    ax.set_ylabel('Frequency', fontsize=16)  # Increase y-axis label font size
# Set a title for the entire figure
fig.suptitle('Histograms of All Variables', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the suptitle
output_path = 'figs/descriptive_histogram.png'
plt.savefig(output_path, dpi=300)
plt.show()


# k-complexity map
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
final_gdf.plot(column="k_complexity", ax=ax, legend=True, cmap='Blues', 
                   legend_kwds={'shrink': 0.5}, 
                   missing_kwds={"color": "lightgrey", "edgecolor": "red", "hatch": "///"})
plt.title('Police Areas Colored by Population Weighted K-Complexity')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
output_path = 'figs/kcomp_map_wholecountry.png'
plt.savefig(output_path, dpi=300)
plt.show()


# plot detail of k-complexity in cape town
minx, miny = 18.50, -34.10
maxx, maxy = 18.78, -33.95
bbox = (minx, miny, maxx, maxy)
# Filter the GeoDataFrame using clip
gdf_cape_town = gpd.clip(gdf_block_info, mask=bbox)
# Plot the GeoDataFrame
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
gdf_cape_town.plot(column='k_complexity', ax=ax, legend=True, cmap='Greens',
                   legend_kwds={'label': "K-Complexity",
                                'orientation': "vertical",
                                'shrink': 0.5,
                                'aspect': 15})
plt.title('Blocks in Cape Town Colored by K-Complexity')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
output_path = 'figs/cape_town_kcomp.png'
plt.savefig(output_path, dpi=300)
plt.show()


# crime rate map
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
# Define the bounding boxes
bbox_main = (16, -35.5, 32, -22) # Whole Country
bbox_cape_town = (18, -34.5, 19.3, -33.7)
bbox_johannesburg = (27.5, -26.5, 28.2, -25.9)
# Create the figure and main axis
fig, ax_main = plt.subplots(1, 1, figsize=(15, 15))
# Plot the main GeoDataFrame with the full map
gpd.clip(final_gdf[final_gdf["landscan_population"] > 1000], mask=bbox_main).plot(
    column='log10_total_crime_per_capita', ax=ax_main, cmap='Blues',
    vmin=0, vmax=0.3, legend=False)
ax_main.set_title('South Africa Police Areas Colored by log10(Crime Rate), averaged 2018-2022', fontsize=40, pad=50)
ax_main.set_xlabel('Longitude', fontsize=30)
ax_main.set_ylabel('Latitude', fontsize=30)
# Add gray bounding boxes for Cape Town and Johannesburg
rect_cape_town = Rectangle((bbox_cape_town[0], bbox_cape_town[1]),
                           bbox_cape_town[2] - bbox_cape_town[0],
                           bbox_cape_town[3] - bbox_cape_town[1],
                           linewidth=2, edgecolor='gray', facecolor='none')
rect_johannesburg = Rectangle((bbox_johannesburg[0], bbox_johannesburg[1]),
                              bbox_johannesburg[2] - bbox_johannesburg[0],
                              bbox_johannesburg[3] - bbox_johannesburg[1],
                              linewidth=2, edgecolor='gray', facecolor='none')
ax_main.add_patch(rect_cape_town)
ax_main.add_patch(rect_johannesburg)
# Create inset axes for Cape Town outside of the main map (left side)
ax_cape_town = inset_axes(ax_main, width="90%", height="90%", loc='center left',
                          bbox_to_anchor=(-0.75, 0.0, 0.45, 0.45), bbox_transform=ax_main.transAxes)
gpd.clip(final_gdf[final_gdf["landscan_population"] > 1000], mask=bbox_cape_town).plot(
    column='log10_total_crime_per_capita', ax=ax_cape_town, cmap='Blues', vmin=0, vmax=0.3, legend=False)
ax_cape_town.set_title('Cape Town', fontsize=30, pad=20)
# Create inset axes for Johannesburg outside of the main map (right side)
ax_johannesburg = inset_axes(ax_main, width="90%", height="90%", loc='center right',
                             bbox_to_anchor=(1.3, 0.5, 0.45, 0.45), bbox_transform=ax_main.transAxes)
gpd.clip(final_gdf[final_gdf["landscan_population"] > 1000], mask=bbox_johannesburg).plot(
    column='log10_total_crime_per_capita', ax=ax_johannesburg, cmap='Blues', vmin=0, vmax=0.3, legend=False)
ax_johannesburg.set_title('Johannesburg', fontsize=30, pad=20)
# Create a thicker shared colorbar for the entire figure with larger font
cax = inset_axes(ax_main, width="5%", height="100%", loc='lower left',
                 bbox_to_anchor=(1.05, 0.0, 0.05, 1.0), bbox_transform=ax_main.transAxes, borderpad=0)
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=0.3))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('K-Complexity', size=20)
cbar.ax.tick_params(labelsize=15)
output_path = 'figs/crime_map_wholecountry.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.show()

# Map for the rest of the four columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
titles = [ "Population Density (Log Scale)", "Building Density (Log Scale)", "Total Road Length", "Average Altitude" ]
for i, (col, title) in enumerate(zip(['log10_landscan_population_density', 'log10_building_density', 'road_length', 'average_altitude'], titles)):
    ax = axes[i//2, i%2]  # Calculate grid position
    cmap = final_gdf.plot(column=col, ax=ax, legend=True, cmap='Blues', 
                          legend_kwds={'shrink': 0.5}, 
                          missing_kwds={"color": "lightgrey", "edgecolor": "red", "hatch": "///"}).get_figure().get_axes()[-1]
    # Set title with larger font size
    ax.set_title(title, fontsize=25)
    ax.axis('off')
    # Adjust colorbar label font size
    cmap.tick_params(labelsize=18)
plt.tight_layout()
output_path = 'figs/descriptive_map.png'
plt.savefig(output_path, dpi=300)
plt.show()





# %% KNN with individual features
#######################
### K-means ANALYSIS ###
#######################

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
features = ['log10_building_density', 'k_complexity', 'log10_landscan_population_density', 'road_length', 'average_altitude']
X = final_gdf[features]
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#######################
# Find the optimal number of clusters using the Elbow method
k_range = range(1, 21)
inertia_values = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Plot for K-Means Clustering', fontsize=18, pad=20)
plt.xlabel('Number of Clusters (k)', fontsize=15)
plt.ylabel('Inertia', fontsize=15)
plt.xticks(k_range, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)  # Ensure this is before plt.savefig()
output_path = 'figs/elbow_plot.png'
plt.savefig(output_path, dpi=300)
plt.show()

#######################
# Choose 7 clusters
kmeans = KMeans(n_clusters=7, random_state=42)
final_gdf['cluster_knn7'] = kmeans.fit_predict(X_scaled)

# VISUALIZE
# Color palette
import matplotlib.patches as mpatches
import textwrap

plt.rcParams.update({
    'font.size': 14,       # general text size
    'axes.titlesize': 16,  # title size
    'axes.labelsize': 14,  # x and y label size
    'legend.fontsize': 12, # legend font size for items
    'xtick.labelsize': 12, # x tick label size
    'ytick.labelsize': 12  # y tick label size
})
# Setup your dataset and variables
unique_clusters = sorted(final_gdf['cluster_knn7'].unique())
colors = plt.cm.tab20(range(len(unique_clusters)))[:len(unique_clusters)]
cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
palette = {str(cluster): color for cluster, color in cluster_colors.items()}
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, var in zip(axes.flatten(), variables):
    sns.boxplot(x='cluster_knn7', y=var, data=final_gdf, ax=ax, palette=palette)
    ax.set_xlabel('Cluster', fontsize=16)
    ax.set_ylabel(var.replace('_', ' ').title(), fontsize=16)
title = 'Box Plot Analysis of All Variables Across Clusters'
wrapped_title = "\n".join(textwrap.wrap(title, width=60))
fig.suptitle(wrapped_title, fontsize=25)
plt.subplots_adjust(top=0.92)  # Adjust the top of the subplot to make room for the title
handles = [mpatches.Patch(color=color, label=f'Cluster {cluster}') for cluster, color in cluster_colors.items()]
legend = fig.legend(handles=handles, title='Clusters', loc='center', bbox_to_anchor=(1.02, 0.5),
                    fontsize=14, title_fontsize=16)
output_path = 'figs/knn7_cluster_boxplots.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()


# map
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
final_gdf['color'] = final_gdf['cluster_knn7'].map(cluster_colors)
final_gdf.plot(color=final_gdf['color'], ax=axs[0])
# Generate custom legend
handles = [mpatches.Patch(color=color, label=f'Cluster {cluster}') for color, cluster in zip(colors, unique_clusters)]
# Title and labels
axs[0].legend(handles=handles, title="Clusters", loc='upper left')
axs[0].set_title('KNN Clustering of South African Police Station Areas')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
# Bar plot for counts
df_count = final_gdf.groupby('cluster_knn7').size().reset_index().rename(columns={0: 'count'})
sns.barplot(x='cluster_knn7', y="count", data=df_count, ax=axs[1], palette=palette)
axs[1].set_xlabel('Cluster')
axs[1].set_ylabel('Count')
axs[1].set_title("Distribution of Area Units Across Clusters")
# save and show plot
output_path = 'figs/knn7_map.png'
plt.savefig(output_path, dpi=300)
plt.show()

# Summary stats
cluster_summary = final_gdf.groupby('cluster_knn7')[features].agg(['mean', 'std'])
print(cluster_summary)

# Check the category (urban / non-urban) of each cluster
intersections = intersections.merge(final_gdf[['cluster_knn7', 'Station']], left_on='police_station', right_on='Station', how='left')
summary_table = intersections.groupby(['cluster_knn5', 'class_urban_hierarchy']).size().unstack(fill_value=0)




# %%
# Check crime numbers by category by cluster
years = ['2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023']
crime_df['average_crime'] = crime_df.loc[:, years].mean(axis=1)
merged_df = pd.merge(final_gdf, crime_df, on='Station')
merged_df['per_capita_crime'] = merged_df['average_crime'] / merged_df['landscan_population']
grouped_df = merged_df.groupby(['cluster_knn7', 'Crime_Category'])['per_capita_crime'].mean().unstack()
# Plot
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(8, 20), sharex=True)
for i, cluster in enumerate(grouped_df.index):
    axes[i].bar(grouped_df.columns, grouped_df.loc[cluster])
    axes[i].set_title(f'Cluster {cluster}')
    axes[i].set_ylim(0, 0.05)
    axes[i].set_ylabel('Per Capita Crime Rate (2018-2022)')
    axes[i].set_xticklabels(grouped_df.columns, rotation=90)
axes[-1].set_xlabel('Crime Category')
plt.tight_layout()
output_path = 'figs/crime_by_cat_by_cluster.png'
plt.savefig(output_path, dpi=300)
plt.show()


# %% Crime and k-complexity correlation by cluster
from scipy.stats import linregress
# Assuming final_gdf is already defined and has the necessary columns
n_clusters = 7
n_cols = 3
n_rows = (n_clusters + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()
# Prepare common legend handles
scatter_handle = mpatches.Patch(color='lightblue', label='Data Points')
line_handle = mpatches.Patch(color='gray', label='Linear Regression')
for cluster_id in range(n_clusters):
    ax = axes[cluster_id]
    cluster_data = final_gdf[final_gdf["cluster_knn7"] == cluster_id]
    x = cluster_data['k_complexity']
    y = cluster_data['log10_total_crime_per_capita']
    if len(x) < 2:
        ax.set_title(f"Cluster {cluster_id} (Insufficient data)")
        continue
    # Normalize
    x_norm = (x - np.mean(x)) / np.std(x)
    y_norm = (y - np.mean(y)) / np.std(y)
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_norm, y_norm)
    line = slope * x_norm + intercept
    t_value = 1.96
    ci = t_value * std_err
    # Scatter and regression line
    ax.scatter(x_norm, y_norm, color='lightblue', alpha=0.7)
    ax.plot(x_norm, line, color='gray')
    # Text box with stats
    textstr = '\n'.join((
        r'$R^2=%.2f$' % (r_value**2, ),
        r'$r=%.2f$' % (r_value, ),
        r'$slope=%.2f \pm %.2f$' % (slope, ci)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.set_title(f"Cluster {cluster_id}")
    ax.set_xlabel("Normalized k_complexity")
    ax.set_ylabel("Normalized log10(total\ncrime per capita)")
for i in range(n_clusters, len(axes)): #hide unused subplots
    fig.delaxes(axes[i])
# Add a single legend outside the grid
fig.legend(handles=[scatter_handle, line_handle], loc='lower center', ncol=2, fontsize=12)
fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
# Save and show
plt.savefig('figs/knn7_all_clusters_lr.png', dpi=300)
plt.show()

# %% T-test of difference in crime between each cluster with cluster 2
from scipy.stats import ttest_ind
# Define Cluster 2 data
cluster_2_data = final_gdf.loc[final_gdf["cluster_knn7"] == 2, "log10_total_crime_per_capita"]

# Store T-test results
results = []

# Loop through all unique clusters other than Cluster 2
for cluster in final_gdf["cluster_knn7"].unique():
    if cluster != 2:
        # Extract data for the current cluster
        other_cluster_data = final_gdf.loc[final_gdf["cluster_knn7"] == cluster, "log10_total_crime_per_capita"]
        
        # Perform independent T-test
        t_stat, p_value = ttest_ind(cluster_2_data, other_cluster_data, equal_var=False)
        
        # Append results formatted to scientific notation with 5 significant digits
        results.append({
            "Comparison": f"Cluster 2 vs Cluster {int(cluster)}",
            "t-Statistic": f"{t_stat:.5e}",
            "p-Value": f"{p_value:.5e}"
        })

# Convert results to a DataFrame
t_test_results_df = pd.DataFrame(results)

# Print results in table format
print("\nT-Test Results (Cluster 2 vs Other Clusters):")
print(t_test_results_df)

# Generate LaTeX code for the table
latex_code = t_test_results_df.to_latex(index=False, escape=False,
                                        caption="T-Test Results: Cluster 2 vs Other Clusters",
                                        label="tab:t_test_results")
print("\nLaTeX Code for Table:")
print(latex_code)
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
final_gdf.assign(Is=lisa.Is).plot(column="Is", cmap="plasma", scheme="quantiles", k=7, 
                                  edgecolor="white", linewidth=0.1, alpha=0.75, legend=True, ax=ax)
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
# Set title to each subplot
for i, ax in enumerate(axs.flatten()): 
    ax.set_axis_off() 
ax.set_title( [ "Local Statistics", "Scatterplot Quadrant", "Statistical Significance", "Moran Cluster Map", ][i], y=0, )
f.tight_layout()
plt.show()



# %%
#https://pysal.org/notebooks/lib/libpysal/weights.html

from libpysal.weights import Queen, Rook, KNN

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Rook neighbors
w_rook = Rook.from_dataframe(final_gdf)
w_rook.transform = "R"  # Row-standardization
lisa_rook = esda.moran.Moran_Local(final_gdf["log10_total_crime_per_capita"], w_rook)
esdaplot.lisa_cluster(lisa_rook, final_gdf, p=0.05, ax=axs[0])
axs[0].set_title('LISA cluster map, Rook Neighbors')

# Plot 2: Queen neighbors
w_queen = Queen.from_dataframe(final_gdf)
w_queen.transform = "R"  # Row-standardization
lisa_queen = esda.moran.Moran_Local(final_gdf["log10_total_crime_per_capita"], w_queen)
esdaplot.lisa_cluster(lisa_queen, final_gdf, p=0.05, ax=axs[1])
axs[1].set_title('LISA cluster map, Queen Neighbors')

# Plot 3: KNN neighbors
w_knn = KNN.from_dataframe(final_gdf, k=5)
w_knn.transform = "R"  # Row-standardization
lisa_knn = esda.moran.Moran_Local(final_gdf["log10_total_crime_per_capita"], w_knn)
esdaplot.lisa_cluster(lisa_knn, final_gdf, p=0.05, ax=axs[2])
axs[2].set_title('LISA cluster map, KNN 5 Neighbors')

# Adjust layout
fig.tight_layout()
output_path = 'figs/hhll_cluster7_combined.png'
plt.savefig(output_path, dpi=300)
plt.show()


# Check within each cluster to see k-complexity vs. crime
from scipy.stats import linregress
# Ensure final_gdf is defined and contains the required columns
final_gdf_cluster2 = final_gdf[final_gdf["cluster_knn7"] == 2]
x = final_gdf_cluster2['k_complexity']
y = final_gdf_cluster2['log10_total_crime_per_capita']
# Normalize both variables
x_normalized = (x - np.mean(x)) / np.std(x)
y_normalized = (y - np.mean(y)) / np.std(y)
# Perform linear regression on normalized data
slope, intercept, r_value, p_value, std_err = linregress(x_normalized, y_normalized)
line = slope * x_normalized + intercept
# Calculate 95% confidence interval for the slope
t_value = 1.96
ci = t_value * std_err
# Create the scatter plot and regression line
fig, ax = plt.subplots()
ax.scatter(x_normalized, y_normalized, label='Data Points')
ax.plot(x_normalized, line, color='red', label='Linear Regression')
# Prepare text box content
textstr = '\n'.join((
    r'$R^2=%.2f$' % (r_value**2, ),
    r'$r=%.2f$' % (r_value, ),
    r'$slope=%.2f \pm %.2f$' % (slope, ci)))
# Place a text box in upper right in axes coordinates
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)
# Add labels and legend
ax.set_xlabel('Normalized k_complexity')
ax.set_ylabel('Normalized log10_total_crime_per_capita')
# Show the plot
output_path = 'figs/knn7_cluster2_lr.png'
plt.savefig(output_path, dpi=300)
plt.show()


# Combine clusters 2-5
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
# Filter clusters 2–5
subset = final_gdf[final_gdf["cluster_knn7"].isin([2, 3, 4, 5])]
x = subset['k_complexity']
y = subset['log10_total_crime_per_capita']
x_norm = (x - x.mean()) / x.std()  # Normalize x
y_norm = (y - y.mean()) / y.std()  # Normalize y

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(x_norm, y_norm)
line = slope * x_norm + intercept  # Use normalized x for the regression line
ci = 1.96 * std_err  # 95% confidence interval
# get right colors
colors = subset['cluster_knn7'].map(lambda c: palette[str(c)])
# Plot
fig, ax = plt.subplots(figsize=(12, 6))  # wider figure
scatter = ax.scatter(x, y_norm, c=colors, alpha=0.7)
ax.plot(x, line, color='gray', alpha=0.5, linewidth=2, label='Regression Line')
# Stats box
textstr = rf'$R^2 = {r_value**2:.2f}$' + '\n' + rf'$slope = {slope:.2f} \pm {ci:.2f}$'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='right', bbox=props)
# Labels and title
ax.set_xlabel('K-complexity', fontsize=14)
ax.set_ylabel('Normalized log10(total_crime_per_capita)', fontsize=14)
ax.set_title('Clusters 2–5: Regression of Normalized Crime on K-complexity', fontsize=16, pad=15)
ax.tick_params(axis='both', labelsize=12)
# Legend outside
handles = [
    plt.Line2D([], [], marker='o', linestyle='', color=cluster_colors[c], label=f'Cluster {c}')
    for c in [2, 3, 4, 5]]
handles.append(plt.Line2D([], [], color='lightblue', alpha=0.5, label='Regression Line'))
ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
# Save and show
plt.tight_layout()
output_path = 'figs/knn7_clusters2to5_combined.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()


# %% K-means sensitivity analysis
# 8 clusters
kmeans = KMeans(n_clusters=8, random_state=42)
final_gdf['cluster_knn8'] = kmeans.fit_predict(X_scaled)
# VISUALIZE
# Color palette
unique_clusters = sorted(final_gdf['cluster_knn8'].unique())
colors = plt.cm.tab20(range(len(unique_clusters)))[:len(unique_clusters)]
cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
# Plot the box plot of all 5 dependent variables, 1 independent variable for all clusters
palette = {str(cluster): color for cluster, color in cluster_colors.items()}
fig, axes = plt.subplots(2, 3, figsize=(18, 12), tight_layout=True)
# Plot each variable in a separate subplot
for ax, var in zip(axes.flatten(), variables):
    sns.boxplot(x='cluster_knn8', y=var, data=final_gdf, ax=ax, palette=palette)
    ax.set_title(f'Box Plot of {var.replace("_", " ").title()} for Each Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel(var.replace('_', ' ').title())
# Add a custom legend for clusters
handles = [mpatches.Patch(color=color, label=f'Cluster {cluster}') for cluster, color in cluster_colors.items()]
# Use `bbox_transform` to position the legend outside the axes on the right side
fig.legend(handles=handles, title='Clusters', loc='center left', bbox_to_anchor=(1.02, 0.5))
output_path = 'figs/knn8_cluster_boxplots.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# One map
final_gdf['color'] = final_gdf['cluster_knn8'].map(cluster_colors)
final_gdf.plot(color=final_gdf['color'], ax=axs[0])
# Generate custom legend
handles = [mpatches.Patch(color=color, label=f'Cluster {cluster}') for color, cluster in zip(colors, unique_clusters)]
# Title and labels
axs[0].legend(handles=handles, title="Clusters", loc='lower right')
axs[0].set_title('KNN Clustering of South African Police Station Areas')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
# Bar plot for counts
df_count = final_gdf.groupby('cluster_knn8').size().reset_index().rename(columns={0: 'count'})
sns.barplot(x='cluster_knn8', y="count", data=df_count, ax=axs[1], palette=palette)
axs[1].set_xlabel('Cluster')
axs[1].set_ylabel('Count')
# save and show plot
output_path = 'figs/knn8_map.png'
plt.savefig(output_path, dpi=300)
plt.show()


# 6 clusters
kmeans = KMeans(n_clusters=6, random_state=42)
final_gdf['cluster_knn6'] = kmeans.fit_predict(X_scaled)
# VISUALIZE
# Color palette
unique_clusters = sorted(final_gdf['cluster_knn6'].unique())
colors = plt.cm.tab20(range(len(unique_clusters)))[:len(unique_clusters)]
cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
# Plot the box plot of all 5 dependent variables, 1 independent variable for all clusters
palette = {str(cluster): color for cluster, color in cluster_colors.items()}
fig, axes = plt.subplots(2, 3, figsize=(18, 12), tight_layout=True)
# Plot each variable in a separate subplot
for ax, var in zip(axes.flatten(), variables):
    sns.boxplot(x='cluster_knn6', y=var, data=final_gdf, ax=ax, palette=palette)
    ax.set_title(f'Box Plot of {var.replace("_", " ").title()} for Each Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel(var.replace('_', ' ').title())
# Add a custom legend for clusters
handles = [mpatches.Patch(color=color, label=f'Cluster {cluster}') for cluster, color in cluster_colors.items()]
# Use `bbox_transform` to position the legend outside the axes on the right side
fig.legend(handles=handles, title='Clusters', loc='center left', bbox_to_anchor=(1.02, 0.5))
output_path = 'figs/knn6_cluster_boxplots.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# One map
final_gdf['color'] = final_gdf['cluster_knn6'].map(cluster_colors)
final_gdf.plot(color=final_gdf['color'], ax=axs[0])
# Generate custom legend
handles = [mpatches.Patch(color=color, label=f'Cluster {cluster}') for color, cluster in zip(colors, unique_clusters)]
# Title and labels
axs[0].legend(handles=handles, title="Clusters", loc='lower right')
axs[0].set_title('KNN Clustering of South African Police Station Areas')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
# Bar plot for counts
df_count = final_gdf.groupby('cluster_knn6').size().reset_index().rename(columns={0: 'count'})
sns.barplot(x='cluster_knn6', y="count", data=df_count, ax=axs[1], palette=palette)
axs[1].set_xlabel('Cluster')
axs[1].set_ylabel('Count')
# save and show plot
output_path = 'figs/knn6_map.png'
plt.savefig(output_path, dpi=300)
plt.show()











# %%

#################################
###### MACHINE LEARNING ##########
#################################

from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Drop rows with NaN values
final_gdf_nona = final_gdf.dropna()
final_gdf_nona = final_gdf_nona.to_crs(epsg=32735)
# Calculate centroids of polygons to obtain representative point (latitude, longitude)
final_gdf_nona['latitude'] = final_gdf_nona['geometry'].centroid.y
final_gdf_nona['longitude'] = final_gdf_nona['geometry'].centroid.x
# Define features
features = ['log10_building_density', 'k_complexity', 'log10_landscan_population_density',
            'road_length', 'average_altitude', 'latitude', 'longitude']
feature_labels = {
    'log10_building_density': 'Log10 Building\nDensity',
    'k_complexity': 'K Complexity',
    'log10_landscan_population_density': 'Log10 LandScan\nPopulation Density',
    'road_length': 'Road Length',
    'average_altitude': 'Average Altitude',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'log10_total_crime_per_capita': 'Log10 Total\nCrime Per Capita'
}
# Get features and target variable
X = final_gdf_nona[features]
y = final_gdf_nona['log10_total_crime_per_capita']
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


###### Correlation Analysis ######
# Create a mapping of feature names to labels
correlation_matrix = final_gdf_nona[features + ['log10_total_crime_per_capita']].corr()
correlation_matrix.rename(index=feature_labels, columns=feature_labels, inplace=True)
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
#plot
plt.figure(figsize=(10, 8))
# Create the heatmap with the mask applied, centering the colormap at 0
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 15}, center=0, mask=mask)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title('Lower Triangle Correlation Matrix', fontsize=20, pad=40)
plt.tight_layout()
plt.savefig('figs/lower_triangle_corr_matrix.png', dpi=300)
plt.show()


# %%
###### KNN ##########
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import pandas as pd

# Check the optimal number of neighbors
neighbor_range = range(1, 21)
mse_values = []

for n_neighbors in neighbor_range:
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

# Evaluate KNN with a chosen number of neighbors
knn = KNeighborsRegressor(n_neighbors=5)  # Change this as needed
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print model evaluation metrics in LaTeX format
metrics_df = pd.DataFrame({
    "Metric": ["Mean Squared Error", "R-squared", "Mean Absolute Error"],
    "Value": [mse, r2, mae]
})

latex_metrics_table = metrics_df.to_latex(index=False, float_format="%.6f", caption="KNN Regression Performance Metrics", label="tab:knn_metrics")
print("\nLaTeX Code for KNN Metrics Table:")
print(latex_metrics_table)

# Feature importances using permutation importance
perm_importance = permutation_importance(knn, X_test, y_test, n_repeats=30, random_state=42)
feature_importances = pd.DataFrame({'Feature': features, 'Importance': perm_importance.importances_mean})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Print feature importance in LaTeX format
latex_feature_importance_table = feature_importances.to_latex(index=False, float_format="%.6f", caption="Feature Importances from Permutation Importance", label="tab:feature_importance")
print("\nLaTeX Code for Feature Importance Table:")
print(latex_feature_importance_table)



# Random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
# Predict on the test set
y_pred_rf = rf.predict(X_test)
# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
# Print Random Forest metrics in LaTeX
metrics_rf_df = pd.DataFrame({
    "Metric": ["Mean Squared Error", "R-squared", "Mean Absolute Error"],
    "Value": [mse_rf, r2_rf, mae_rf]
})
latex_metrics_rf_table = metrics_rf_df.to_latex(index=False, float_format="%.6f",
                                                caption="Random Forest Regression Metrics",
                                                label="tab:rf_metrics")
print("\nLaTeX Code for Random Forest Metrics:")
print(latex_metrics_rf_table)
# Feature importances from Random Forest
feature_importances_rf = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
feature_importances_rf = feature_importances_rf.sort_values(by="Importance", ascending=False)

# Print Random Forest feature importances in LaTeX
latex_feature_importance_rf_table = feature_importances_rf.to_latex(index=False, float_format="%.6f",
                                                                    caption="Feature Importances from Random Forest",
                                                                    label="tab:rf_feature_importance")
print("\nLaTeX Code for Random Forest Feature Importance Table:")
print(latex_feature_importance_rf_table)



# %% Plot a tree
from sklearn.tree import plot_tree

# Visualize the first few layers of the first tree in the Random Forest with larger text
plt.figure(figsize=(20, 10))
plot_tree(rf.estimators_[0], feature_names=features, filled=True, rounded=True, fontsize=15, max_depth=2)  # Adjusted fontsize
plt.title("Partial Decision Tree from the Random Forest (First 2 Layers)", fontsize=24)  # Title text size increased
plt.savefig("figs/sample_tree.png")
plt.show()

# %%

## more analysis ##

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize Feature Importances: KNN
def plot_feature_importances(feature_importances, model_name):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
    plt.title(f'{model_name} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

print("\nVisualizing KNN Feature Importances:")
plot_feature_importances(feature_importances, 'KNN')

# Visualize Feature Importances: Random Forest
print("\nVisualizing Random Forest Feature Importances:")
plot_feature_importances(feature_importances_rf, 'Random Forest')




# %%

# Partial Dependence Plot using RandomForest
from sklearn.inspection import partial_dependence, plot_partial_dependence

def plot_partial_dependence_rf(model, features, feature_names, X_train):
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_partial_dependence(model, X_train, features, feature_names=feature_names, ax=ax)
    plt.show()

print("\nPartial Dependence Plot Analysis:")
plot_partial_dependence_rf(rf, [0, 1, 2, 3, 4], features, X_train)


# Permutation Importance with Interaction
def plot_permutation_importance_with_interaction(model, X_test, y_test):
    from sklearn.inspection import permutation_importance

    # Perform permutation importance with interactions
    perm_importance_interactions = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    feature_importance_interactions = pd.DataFrame({
        'Feature': features,
        'Importance': perm_importance_interactions.importances_mean
    }).sort_values(by='Importance', ascending=False)

    print("\nPermutation Importance with Interaction:")
    plot_feature_importances(feature_importance_interactions, 'With Interaction')

plot_permutation_importance_with_interaction(rf, X_test, y_test)