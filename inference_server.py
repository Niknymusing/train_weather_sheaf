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
import json
import socket
import io
import base64
from snet import sdk
from photrek_call import retry_with_delay, SnetConfig #call_photrek



USE_PHOTREK = False

# Add the repository root to the Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(repo_root, 'FourCastNet'))

from setup_inference import setup_model






#output_distribution = ProbabilityProjection()


# Set up the AFNO model
#backbone_checkpoint = os.path.expanduser('~/Downloads/backbone.ckpt')
#model = setup_model() #, backbone_checkpoint)

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


def call_photrek(p, q):
    """
    Call the Photrek service with enhanced logging to debug response format and values.
    
    Args:
        p: First probability distribution (tensor or array-like)
        q: Second probability distribution (tensor or array-like)
        
    Returns:
        dict: Dictionary with keys 'accuracy', 'decisiveness', 'robustness'
    """
    # Log input tensors
    print("\n=== Photrek API Call Debug Log ===")
    print("\nInput Distributions:")
    print(f"p shape: {p.shape if isinstance(p, torch.Tensor) else 'not tensor'}")
    print(f"q shape: {q.shape if isinstance(q, torch.Tensor) else 'not tensor'}")
    
    # Ensure inputs are tensors
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    else:
        p = p.clone().detach()
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32)
    else:
        q = q.clone().detach()
    
    # Log normalized tensor values
    print("\nNormalized Input Values:")
    print(f"p: {p}")
    print(f"q: {q}")
    
    # Ensure numerical stability
    eps = 1e-10
    p = torch.clamp(p, min=eps)
    p = p / p.sum()
    q = torch.clamp(q, min=eps)
    q = q / q.sum()
    
    try:
        # Create configuration
        config_obj = SnetConfig()
        print("\nConfiguration:")
        print(f"Network: {config_obj.get('network')}")
        print(f"Identity: {config_obj.get('identity_type')}:{config_obj.get('identity_name')}")
        
        # Initialize SDK
        def init_sdk():
            return sdk.SnetSDK(config_obj)
        snet_sdk = retry_with_delay(init_sdk)
        print("\nSDK initialized successfully")
        
        # Create service client
        def create_client():
            return snet_sdk.create_service_client(
                org_id="Photrek",
                service_id="risk-aware-assessment",
                group_name="default_group"
            )
        service_client = retry_with_delay(create_client)
        print("\nService client created successfully")
        
        # Create CSV data
        buffer = io.StringIO()
        p_np = p.numpy()
        q_np = q.numpy()
        num_classes = len(p_np)
        
        # Write distributions
        probs = [str(x) for x in p_np]
        buffer.write(','.join(probs + ['1']) + '\n')
        probs = [str(x) for x in q_np]
        buffer.write(','.join(probs + ['2']) + '\n')
        
        csv_content = buffer.getvalue()
        print("\nGenerated CSV content:")
        print(csv_content)
        
        csv_base64 = base64.b64encode(csv_content.encode()).decode()
        input_str = f"2,{num_classes+1},{csv_base64}"
        
        print("\nAPI Input String Format:")
        print(f"Rows: 2")
        print(f"Columns: {num_classes+1}")
        print(f"Base64 length: {len(csv_base64)}")
        
        def call_service():
            print("\nCalling Photrek API...")
            response = service_client.call_rpc(
                rpc_name="adr",
                message_class="InputString",
                s=input_str
            )
            print("\nRaw API Response:")
            print(f"Response type: {type(response)}")
            print(f"Response attributes: {dir(response)}")
            print(f"Response values: {response}")
            return response
            
        response = retry_with_delay(call_service)
        
        # Log complete response details
        print("\nProcessed Response:")
        if hasattr(response, 'a'):
            metrics = {
                "accuracy": float(response.a),
                "decisiveness": float(response.d),
                "robustness": float(response.r)
            }
            print("Metrics computed successfully:")
            print(f"Accuracy: {metrics['accuracy']}")
            print(f"Decisiveness: {metrics['decisiveness']}")
            print(f"Robustness: {metrics['robustness']}")
            
            # Compare with local implementation
            print("\nComparing with local implementation:")
            local_metrics = compute_metrics(p, q)
            print(f"Local metrics: {local_metrics}")
            print(f"Difference in accuracy: {abs(metrics['accuracy'] - local_metrics['accuracy'])}")
            print(f"Difference in decisiveness: {abs(metrics['decisiveness'] - local_metrics['decisiveness'])}")
            print(f"Difference in robustness: {abs(metrics['robustness'] - local_metrics['robustness'])}")
            
            return metrics
        else:
            print("\nERROR: Unexpected response format")
            print(f"Available attributes: {dir(response)}")
            raise Exception("Unexpected response format from service")
            
    except Exception as e:
        print(f"\nERROR in call_photrek: {str(e)}")
        print("Exception details:")
        import traceback
        traceback.print_exc()
        raise


def verify_metrics_compatibility(api_metrics, local_metrics, tolerance=1e-6):
    """
    Verify that API and local metrics are compatible within tolerance.
    """
    differences = {
        key: abs(api_metrics[key] - local_metrics[key])
        for key in ['accuracy', 'decisiveness', 'robustness']
    }
    
    is_compatible = all(diff <= tolerance for diff in differences.values())
    
    print("\nMetrics Compatibility Check:")
    print(f"Using tolerance: {tolerance}")
    for key, diff in differences.items():
        print(f"{key}: difference = {diff}")
    print(f"Compatible: {is_compatible}")
    
    return is_compatible


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




def find_free_port(start_port=5000, max_port=5010):
    """Find a free port to use by testing a range of ports."""
    import socket
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise OSError("No free ports found in range")



from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)
    

def validate_reference_distribution(dist):
    """Validate and normalize reference distribution."""
    try:
        dist = np.array(dist, dtype=np.float32)
        if len(dist) != 1000:
            raise ValueError("Reference distribution must have length 1000")
        if np.any(dist < 0):
            raise ValueError("Reference distribution cannot contain negative values")
        if np.all(dist == 0):
            raise ValueError("Reference distribution cannot be all zeros")
        return dist / np.sum(dist)
    except Exception as e:
        raise ValueError(f"Invalid reference distribution: {str(e)}")
    


def process_weather_request(latitude, longitude, timezone, client_reference_distribution=None):
    
    try:
        # 1-5. [Previous data fetching and processing code remains unchanged]
        weather_df = fetch_weather_data_until_complete(latitude, longitude, timezone)
        weather_df["latitude"] = latitude
        weather_df["longitude"] = longitude
        calculate_wind_components(weather_df)
        
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
                weather_df[col] = np.nan

        params = Params()
        era5_dataset = ERA5Dataset(params, weather_df)
        data_loader = DataLoader(era5_dataset, batch_size=1, shuffle=False)

        # 6. Model inference
        batch_inp, _ = next(iter(data_loader))
        with torch.no_grad():
            model_output = model(batch_inp[0].unsqueeze(0))
        # Squeeze and normalize the model output
        model_output = model_output.squeeze()
        model_output = model_output / model_output.sum()

        # 7. Process reference distribution
        try:
            if client_reference_distribution is not None:
                reference_dist = validate_reference_distribution(client_reference_distribution)
                reference_dist = np.array(reference_dist, dtype=np.float32)
                reference_dist = reference_dist / reference_dist.sum()
            else:
                # Generate random reference distribution if none provided
                reference_dist = np.random.rand(len(model_output))
                reference_dist = reference_dist / reference_dist.sum()
            # Convert to torch tensor
            reference_dist = torch.tensor(reference_dist, dtype=torch.float32)
        except ValueError as e:
            print(f"Reference distribution error: {str(e)}")
            # Fallback to random distribution
            random_values = torch.rand(len(model_output))
            reference_dist = random_values / random_values.sum()
            # Convert to torch tensor
            reference_dist = torch.tensor(reference_dist, dtype=torch.float32)

        # 8. Compute metrics
        if USE_PHOTREK:
            try:
                # Ensure model_output and reference_dist have the same length
                if len(model_output) != len(reference_dist):
                    raise ValueError("Model output and reference distribution must have the same length")

                metrics = call_photrek(model_output, reference_dist)
            except Exception as e:
                print('Photrek service error:', str(e))
                metrics = compute_metrics(model_output, reference_dist)
        else:
            metrics = compute_metrics(model_output, reference_dist)

        # 9-10. Prepare response
        weather_dict = weather_df.copy()
        weather_dict['date'] = weather_dict['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        response = {
            'weather_data': weather_dict.to_dict(orient='records'),
            'model_output': model_output.cpu().numpy().tolist(),
            'metrics': metrics
        }

        return response

    except Exception as e:
        print(f"Error in process_weather_request: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def main():
    global model
    model = setup_model()
    
    host = '0.0.0.0'
    port = find_free_port()
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Server listening on port {port}...")
        
        while True:
            try:
                client_socket, address = server_socket.accept()
                print(f"Connection from {address}")
                
                # Receive data length and data together
                length_data = client_socket.recv(1024).decode().strip()
                if not length_data:
                    raise ValueError("No data received")
                
                # Split length_data at the first newline
                if '\n' in length_data:
                    length_str, rest = length_data.split('\n', 1)
                    data_length = int(length_str)
                    data = rest.encode()
                else:
                    # If no newline found, the data might be incomplete
                    length_str = length_data
                    data_length = int(length_str)
                    data = b""
                
                # Now read the rest of the data if necessary
                while len(data) < data_length:
                    chunk = client_socket.recv(min(4096, data_length - len(data)))
                    if not chunk:
                        break
                    data += chunk
                
                request = json.loads(data.decode())
                print(f"Processing request for coordinates: {request['latitude']}, {request['longitude']}")
                
                result = process_weather_request(
                    latitude=request['latitude'],
                    longitude=request['longitude'],
                    timezone=request['timezone'],
                    client_reference_distribution=request.get('reference_distribution')
                )
                
                # Send response length first, then data
                response_data = json.dumps(result, cls=DateTimeEncoder).encode()
                client_socket.send(str(len(response_data)).encode() + b'\n')
                
                # Send response in chunks
                total_sent = 0
                while total_sent < len(response_data):
                    sent = client_socket.send(response_data[total_sent:total_sent + 4096])
                    if sent == 0:
                        raise RuntimeError("Socket connection broken")
                    total_sent += sent
                
                print("Response sent successfully")
                client_socket.close()
                
            except Exception as e:
                print(f"Error processing request: {e}")
                import traceback
                traceback.print_exc()
                if 'client_socket' in locals():
                    client_socket.close()
                continue
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server_socket.close()




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