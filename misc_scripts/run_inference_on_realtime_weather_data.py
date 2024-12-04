import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.interpolate import griddata
import importlib
import sys
import os
from networks.probability_projection import ProbabilityProjection
from photrek_algo import compute_metrics
from copernicus_data import fetch_era5_data

# Add the repository root to the Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(repo_root, 'FourCastNet'))

from setup_inference import setup_model, params 

#output_distribution = ProbabilityProjection()


# Set up the AFNO model
backbone_checkpoint = os.path.expanduser('~/Downloads/backbone.ckpt')
model = setup_model(params, backbone_checkpoint)

random_values = torch.rand(1000)
reference_distribution = random_values / random_values.sum()


def fetch_weather_data(latitude, longitude, timezone):
    """
    Fetch weather data using OpenMeteo API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        timezone (str): Timezone (e.g., "GMT+0").

    Returns:
        pd.DataFrame: Processed hourly weather data as a DataFrame.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "precipitation", "pressure_msl", "surface_pressure",
            "wind_speed_10m", "wind_direction_10m",
            "temperature_1000hPa", "geopotential_height_1000hPa", "wind_speed_1000hPa", "wind_direction_1000hPa",
            "temperature_850hPa", "geopotential_height_850hPa", "relative_humidity_850hPa",
            "wind_speed_850hPa", "wind_direction_850hPa",
            "temperature_500hPa", "geopotential_height_500hPa", "relative_humidity_500hPa",
            "wind_speed_500hPa", "wind_direction_500hPa",
            "geopotential_height_50hPa", "total_column_integrated_water_vapour"
        ],
        "timezone": timezone
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    variable_names = [
        "temperature_2m", "relative_humidity_2m", "precipitation", "pressure_msl", "surface_pressure",
        "wind_speed_10m", "wind_direction_10m",
        "temperature_1000hPa", "geopotential_height_1000hPa", "wind_speed_1000hPa", "wind_direction_1000hPa",
        "temperature_850hPa", "geopotential_height_850hPa", "relative_humidity_850hPa",
        "wind_speed_850hPa", "wind_direction_850hPa",
        "temperature_500hPa", "geopotential_height_500hPa", "relative_humidity_500hPa",
        "wind_speed_500hPa", "wind_direction_500hPa",
        "geopotential_height_50hPa", "total_column_integrated_water_vapour"
    ]
    for i, var_name in enumerate(variable_names):
        hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data=hourly_data)

def fetch_weather_data_until_complete(latitude, longitude, timezone, max_retries=5):
    """
    Fetch weather data repeatedly until all required variables are present or retries are exhausted.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        timezone (str): Timezone (e.g., "GMT+0").
        max_retries (int): Maximum number of retries.

    Returns:
        pd.DataFrame: Complete weather data as a DataFrame.

    Raises:
        ValueError: If required variables are missing after retries.
    """
    required_columns = [
        "temperature_2m", "relative_humidity_2m", "precipitation", "pressure_msl", "surface_pressure",
        "wind_speed_10m", "wind_direction_10m",
        "temperature_1000hPa", "geopotential_height_1000hPa", "wind_speed_1000hPa", "wind_direction_1000hPa",
        "temperature_850hPa", "geopotential_height_850hPa", "relative_humidity_850hPa",
        "wind_speed_850hPa", "wind_direction_850hPa",
        "temperature_500hPa", "geopotential_height_500hPa", "relative_humidity_500hPa",
        "wind_speed_500hPa", "wind_direction_500hPa",
        "geopotential_height_50hPa", "total_column_integrated_water_vapour"
    ]

    for attempt in range(max_retries):
        print(f"Fetching weather data (attempt {attempt + 1}/{max_retries})...")
        weather_df = fetch_weather_data(latitude, longitude, timezone)
        print("Fetched columns:", weather_df.columns.tolist())

        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in weather_df.columns]
        if not missing_columns:
            print("All required variables fetched successfully.")
            return weather_df
        else:
            print(f"Warning: Missing columns: {missing_columns}")
        
        if attempt < max_retries - 1:
            print("Retrying...")
    
    raise ValueError(f"Failed to fetch all required variables after {max_retries} retries. Missing: {missing_columns}")



def calculate_wind_components(weather_df):
    """Calculate wind components for all levels with correct naming"""
    for level in ["10m", "1000hPa", "850hPa", "500hPa"]:
        speed_col = f"wind_speed_{level}"
        dir_col = f"wind_direction_{level}"
        
        if speed_col in weather_df and dir_col in weather_df:
            speeds = weather_df[speed_col]
            directions = weather_df[dir_col]
            
            radians = np.radians(directions)
            U = -speeds * np.sin(radians)
            V = -speeds * np.cos(radians)
            
            # Use correct naming for 10m level
            if level == "10m":
                weather_df["U10"] = U
                weather_df["V10"] = V
            else:
                weather_df[f"U_{level}"] = U
                weather_df[f"V_{level}"] = V


class ERA5Dataset(Dataset):
    """PyTorch Dataset formatted like ERA5 data for NVIDIA FourCastNet."""
    def __init__(self, params, weather_data):
        """
        Args:
            params: Model parameters containing `N_in_channels` and `N_out_channels`.
            weather_data (pd.DataFrame or np.ndarray): DataFrame or reshaped data array.
        """
        self.params = params
        self.img_shape_x = 720  # ERA5 grid size
        self.img_shape_y = 1440
        self.n_in_channels = params.N_in_channels
        self.n_out_channels = params.N_out_channels

        # Define the required input/output variables
        self.input_variables = [
            "U10", "V10", "T2m", "sp", "mslp", 
            "U_1000hPa", "V_1000hPa", "Z_1000hPa", 
            "T_850hPa", "U_850hPa", "V_850hPa", "Z_850hPa", "RH_850hPa",
            "T_500hPa", "U_500hPa", "V_500hPa", "Z_500hPa", "RH_500hPa",
            "Z_50hPa", "T CW V"
        ]
        self.target_variables = self.input_variables  # Assume targets are the same as inputs

        # Handle reshaped or interpolated data
        if isinstance(weather_data, pd.DataFrame):
            self.data = self.interpolate_to_era5_grid(weather_data)
        else:
            self.data = weather_data  # Already reshaped/interpolated

    def interpolate_to_era5_grid(self, data):
        """
        Interpolates the weather data to match the ERA5 grid size (720x1440).
        If data corresponds to a single location, return the data without interpolation.
        Args:
            data (pd.DataFrame): Original weather data.
        Returns:
            np.ndarray: Interpolated or reshaped data.
        """
        # Check if latitude and longitude are constant (single location)
        if data["latitude"].nunique() == 1 and data["longitude"].nunique() == 1:
            print("Single location data detected. Skipping interpolation.")
            # Reshape single-location data to match ERA5 grid dimensions
            reshaped_data = {}
            for var in self.input_variables:
                reshaped_data[var] = np.full((self.img_shape_x, self.img_shape_y), data[var].mean())
            return np.stack([reshaped_data[var] for var in self.input_variables], axis=0)

        # Proceed with interpolation for spatial data
        print("Performing interpolation to ERA5 grid.")
        era5_grid_lat = np.linspace(-90, 90, self.img_shape_x)
        era5_grid_lon = np.linspace(-180, 180, self.img_shape_y)
        era5_grid = np.meshgrid(era5_grid_lon, era5_grid_lat)

        interpolated_data = {}
        for var in self.input_variables:
            points = np.column_stack((data["longitude"], data["latitude"]))
            values = data[var]
            interpolated = griddata(points, values, (era5_grid[1], era5_grid[0]), method="linear")
            interpolated_data[var] = interpolated

        # Convert interpolated data into a 3D tensor for easier slicing
        interpolated_array = np.stack([interpolated_data[var] for var in self.input_variables], axis=0)
        return interpolated_array

    def __len__(self):
        # For ERA5, the dataset size is determined by the time dimension
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Generate input and target tensors with correct shapes.
        """
        # Extract the input and target data
        inp = torch.tensor(self.data[:, :, :], dtype=torch.float32)
        tar = torch.tensor(self.data[:, :, :], dtype=torch.float32)

        return inp, tar


# Define the Params class globally to make it pickleable

class Params:
    N_in_channels = 21
    N_out_channels = 21


def main():

    
    # Input: Geoposition and timezone
    latitude = 59.3293  # Stockholm latitude
    longitude = 18.0686  # Stockholm longitude
    timezone = "Europe/Stockholm"

    # Fetch weather data until all variables are present
    #weather_df = fetch_era5_data(latitude, longitude, timezone)#fetch_weather_data_until_complete(latitude, longitude, timezone)
    weather_df = fetch_weather_data_until_complete(latitude, longitude, timezone)


    # Add latitude and longitude columns for interpolation
    weather_df["latitude"] = latitude
    weather_df["longitude"] = longitude

    # First calculate wind components
    calculate_wind_components(weather_df)

    # Rename variables to match required names
    rename_mapping = {
        "temperature_2m": "T2m",
        "surface_pressure": "sp",
        "pressure_msl": "mslp",
        "geopotential_height_1000hPa": "Z_1000hPa",
        "temperature_850hPa": "T_850hPa",
        "geopotential_height_850hPa": "Z_850hPa",
        "relative_humidity_850hPa": "RH_850hPa",
        "temperature_500hPa": "T_500hPa",
        "geopotential_height_500hPa": "Z_500hPa",
        "relative_humidity_500hPa": "RH_500hPa",
        "geopotential_height_50hPa": "Z_50hPa",
        "total_column_integrated_water_vapour": "T CW V"
    }
    weather_df.rename(columns=rename_mapping, inplace=True)

    

    # Verify all required variables are present
    required_columns = [
        "U10", "V10", "T2m", "sp", "mslp",
        "U_1000hPa", "V_1000hPa", "Z_1000hPa",
        "T_850hPa", "U_850hPa", "V_850hPa", "Z_850hPa", "RH_850hPa",
        "T_500hPa", "U_500hPa", "V_500hPa", "Z_500hPa", "RH_500hPa",
        "Z_50hPa", "T CW V"
    ]
    missing_columns = [col for col in required_columns if col not in weather_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns after processing: {missing_columns}")
        for col in missing_columns:
            # Add missing columns with placeholder values (e.g., NaN or zero)
            weather_df[col] = np.nan

    # Save DataFrame for debugging/inspection
    weather_df.to_csv("weather_data_debug.csv", index=False)

    print("Weather data with calculated variables saved to 'weather_data_debug.csv'.")
    print(weather_df)
    # Prepare ERA5-compatible PyTorch Dataset
    print("Creating PyTorch Dataset...")
    params = Params()
    era5_dataset = ERA5Dataset(params, weather_df)
    # Save the dataset to a .pt file
    dataset_file = "weather_data.pt"
    torch.save(era5_dataset, dataset_file)
    print(f"PyTorch dataset saved as '{dataset_file}'.")

    # Example DataLoader usage
    data_loader = DataLoader(era5_dataset, batch_size=8, shuffle=True)
    for batch_inp, batch_tar in data_loader:
        
        print("Batch input shape:", batch_inp.shape)  # Expected: [8, 21, 720, 1440]
        #print(batch_inp[0].unsqueeze(0))
        out = model(batch_inp[0].unsqueeze(0))
        #probabilities = output_distribution(out, 0, 0)
        print('output distribution: ', out)
        print(compute_metrics(out, reference_distribution))
        print("Batch target shape:", batch_tar.shape)  # Expected: [8, 21, 720, 1440]
        break




# SERVER APP:
# setup: download model weights from S3 bucket (if not downloaded), load model weights, instantiate model
# wait for call with [coordinates, time_stamp] 
# retrieve current weather data from open-meteo api
# construct torch tensors of past time-window of hourly data 
# 
# and input to model to make predictions for each step one step ahead
# convert predictions time series to sequence of probability distribution outputs
# call singularity net service to calculate metrics for each past step, and take average of metrics scores
# 
#
# CLIENT APP:
# input location coordinate (automatically fetch time zone)
# visualise historic data, visualise prediction, 
# optionally visualise the models accuracy, decisiveness, robustness
# 
# TRAINER APP:
# setup: download model weights from S3 bucket (if not downloaded),
# download data and construct dataset
# train model locally
# upload trained model to server and get a unique model ID
# 
#

if __name__ == "__main__":
    main()
