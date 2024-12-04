import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import time

class ERA5Fetcher:
    """Fetches ERA5 reanalysis data from Copernicus Climate Data Store."""
    
    def __init__(self, verbose=True):
        """
        Initialize the ERA5 fetcher with credential verification and connection testing.
        
        Args:
            verbose (bool): Whether to print status messages
        """
        self.verbose = verbose
        self._setup_client()
        
    def _print(self, message):
        """Print if verbose mode is on"""
        if self.verbose:
            print(message)
            
    def _setup_client(self):
        """Setup and verify CDS API client with credentials"""
        self._print("\nInitializing ERA5 data fetcher...")
        
        # Verify environment variables
        url = os.getenv('CDSAPI_URL')
        key = os.getenv('CDSAPI_KEY')
        
        if not url or not key:
            self._print("⚠️  CDS API credentials not found in environment variables!")
            self._print("Please ensure you have added to ~/.zshrc:")
            self._print("export CDSAPI_URL='https://cds.climate.copernicus.eu/api/v2'")
            self._print("export CDSAPI_KEY='<your-uid>:<your-api-key>'")
            self._print("\nAnd run: source ~/.zshrc")
            raise ValueError("Missing CDS API credentials in environment")
            
        try:
            self._print("Initializing CDS API client...")
            self.client = cdsapi.Client()
            self._test_connection()
        except Exception as e:
            self._print(f"\n❌ Error initializing CDS API client: {str(e)}")
            raise

    def _test_connection(self):
        """Test the API connection and license acceptance with a minimal request"""
        self._print("\nTesting CDS API connection...")
        
        try:
            # Small test request
            test_file = 'era5_test.nc'
            self.client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': '2m_temperature',
                    'year': '2024',
                    'month': '01',
                    'day': '01',
                    'time': '12:00',
                    'format': 'netcdf',
                    'area': [90, -180, -90, 180],
                },
                test_file
            )
            
            # Verify the file was created
            if os.path.exists(test_file):
                os.remove(test_file)
                self._print("✅ CDS API connection successful!")
            else:
                raise FileNotFoundError("Test file was not created")
                
        except Exception as e:
            self._print("\n❌ Connection test failed!")
            self._print("\nPlease ensure:")
            self._print("1. You have accepted the ERA5 licenses at:")
            self._print("   https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download#manage-licences")
            self._print("   https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences")
            self._print("2. Your API credentials are correct")
            self._print("3. You're using an ECMWF account (not an old CDS account)")
            raise
    
    def fetch_pressure_levels_data(self, year, month, day, hour, latitude, longitude):
        """
        Fetch ERA5 pressure levels data.
        """
        self._print(f"\nFetching pressure levels data for {year}-{month:02d}-{day:02d} {hour:02d}:00...")
        
        pressure_levels_request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential',                 # For height levels
                'relative_humidity',            # For RH at pressure levels
                'temperature',                  # For temperature at pressure levels
                'u_component_of_wind',          # For wind U at pressure levels
                'v_component_of_wind',          # For wind V at pressure levels
            ],
            'pressure_level': [
                '1000', '850', '500', '50',    # Required pressure levels
            ],
            'year': str(year),
            'month': str(month).zfill(2),
            'day': str(day).zfill(2),
            'time': f'{str(hour).zfill(2)}:00',
            'area': [
                latitude + 0.25, longitude - 0.25,
                latitude - 0.25, longitude + 0.25,
            ],
        }
        
        pressure_file = f'era5_pressure_levels_{year}{month:02d}{day:02d}_{hour:02d}.nc'
        self.client.retrieve('reanalysis-era5-pressure-levels', pressure_levels_request, pressure_file)
        self._print(f"✅ Pressure levels data saved to {pressure_file}")
        return pressure_file

    def fetch_single_levels_data(self, year, month, day, hour, latitude, longitude):
        """
        Fetch ERA5 single level data.
        """
        self._print(f"\nFetching single levels data for {year}-{month:02d}-{day:02d} {hour:02d}:00...")
        
        single_levels_request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_temperature',
                '2m_relative_humidity',
                'mean_sea_level_pressure',
                'surface_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'total_column_water_vapour',
                'total_precipitation',
            ],
            'year': str(year),
            'month': str(month).zfill(2),
            'day': str(day).zfill(2),
            'time': f'{str(hour).zfill(2)}:00',
            'area': [
                latitude + 0.25, longitude - 0.25,
                latitude - 0.25, longitude + 0.25,
            ],
        }
        
        single_file = f'era5_single_levels_{year}{month:02d}{day:02d}_{hour:02d}.nc'
        self.client.retrieve('reanalysis-era5-single-levels', single_levels_request, single_file)
        self._print(f"✅ Single levels data saved to {single_file}")
        return single_file

    def process_data(self, pressure_file, single_file):
        """Process the downloaded ERA5 data files into a pandas DataFrame."""
        self._print("\nProcessing downloaded data...")
        
        # Read NetCDF files
        ds_pressure = xr.open_dataset(pressure_file)
        ds_single = xr.open_dataset(single_file)
        
        # Convert geopotential to geopotential height (divide by g = 9.80665)
        g = 9.80665
        
        # Initialize dictionary for DataFrame
        data = {}
        
        # Process single level variables
        data['temperature_2m'] = ds_single['t2m'].values[0, 0, 0] - 273.15  # Convert K to °C
        data['relative_humidity_2m'] = ds_single['r2'].values[0, 0, 0]
        data['pressure_msl'] = ds_single['msl'].values[0, 0, 0] / 100  # Convert Pa to hPa
        data['surface_pressure'] = ds_single['sp'].values[0, 0, 0] / 100  # Convert Pa to hPa
        
        # Calculate 10m wind speed and direction from components
        u10 = ds_single['u10'].values[0, 0, 0]
        v10 = ds_single['v10'].values[0, 0, 0]
        data['wind_speed_10m'] = np.sqrt(u10**2 + v10**2)
        data['wind_direction_10m'] = np.mod(180 + np.degrees(np.arctan2(u10, v10)), 360)
        
        # Process pressure level variables
        for level in ['1000', '850', '500', '50']:
            idx = ds_pressure.level == float(level)
            
            if level != '50':  # All variables except for 50hPa
                data[f'temperature_{level}hPa'] = ds_pressure['t'].values[0, idx, 0, 0][0] - 273.15
                
                u = ds_pressure['u'].values[0, idx, 0, 0][0]
                v = ds_pressure['v'].values[0, idx, 0, 0][0]
                data[f'wind_speed_{level}hPa'] = np.sqrt(u**2 + v**2)
                data[f'wind_direction_{level}hPa'] = np.mod(180 + np.degrees(np.arctan2(u, v)), 360)
                
                if level in ['850', '500']:  # RH only for 850 and 500
                    data[f'relative_humidity_{level}hPa'] = ds_pressure['r'].values[0, idx, 0, 0][0]
            
            # Geopotential height for all levels
            data[f'geopotential_height_{level}hPa'] = ds_pressure['z'].values[0, idx, 0, 0][0] / g
        
        # Total column water vapor
        data['total_column_integrated_water_vapour'] = ds_single['tcwv'].values[0, 0, 0]
        
        # Clean up
        ds_pressure.close()
        ds_single.close()
        os.remove(pressure_file)
        os.remove(single_file)
        
        self._print("✅ Data processing complete!")
        return pd.DataFrame([data])


def fetch_era5_data(latitude, longitude, timezone):
    """
    Main function to fetch and process ERA5 data for a specific location.
    
    Args:
        latitude (float): Latitude
        longitude (float): Longitude
        timezone (str): Timezone string (not used for ERA5 but kept for compatibility)
        
    Returns:
        pd.DataFrame: Processed weather data
    """
    # Initialize ERA5 fetcher
    fetcher = ERA5Fetcher()
    
    # Get current time in UTC
    current_time = pd.Timestamp.utcnow()
    
    # ERA5 data has about 5 days delay, so we fetch data from 6 days ago
    target_time = current_time - pd.Timedelta(days=6)
    
    # Fetch data
    pressure_file = fetcher.fetch_pressure_levels_data(
        target_time.year, target_time.month, target_time.day, 
        target_time.hour, latitude, longitude
    )
    
    single_file = fetcher.fetch_single_levels_data(
        target_time.year, target_time.month, target_time.day,
        target_time.hour, latitude, longitude
    )
    
    # Process data
    return fetcher.process_data(pressure_file, single_file)

# Add a test function
def test_era5_fetcher():
    """Test the ERA5Fetcher functionality"""
    try:
        # Test location (Stockholm)
        latitude = 59.3293
        longitude = 18.0686
        timezone = "Europe/Stockholm"
        
        print("\nTesting ERA5 data fetching for Stockholm...")
        df = fetch_era5_data(latitude, longitude, timezone)
        
        print("\nRetrieved data columns:")
        for col in df.columns:
            print(f"- {col}: {df[col].values[0]}")
            
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_era5_fetcher()



"""import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import os


# Set CDS API credentials at module level
#os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'
#os.environ['CDSAPI_KEY'] = '01b3b7dc-3194-4e89-93cc-5249d342c7a9'  # Replace with your actual API key



class ERA5Fetcher:
    
    
    def __init__(self):
        
        self.client = cdsapi.Client()
        
    def fetch_pressure_levels_data(self, year, month, day, hour, latitude, longitude):
        
        pressure_levels_request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential',                 # For height levels
                'relative_humidity',            # For RH at pressure levels
                'temperature',                  # For temperature at pressure levels
                'u_component_of_wind',          # For wind U at pressure levels
                'v_component_of_wind',          # For wind V at pressure levels
            ],
            'pressure_level': [
                '1000', '850', '500', '50',    # Required pressure levels
            ],
            'year': str(year),
            'month': str(month).zfill(2),
            'day': str(day).zfill(2),
            'time': f'{str(hour).zfill(2)}:00',
            'area': [
                latitude + 0.25, longitude - 0.25,
                latitude - 0.25, longitude + 0.25,
            ],
        }
        
        # Download pressure levels data
        pressure_file = f'era5_pressure_levels_{year}{month:02d}{day:02d}_{hour:02d}.nc'
        self.client.retrieve('reanalysis-era5-pressure-levels', pressure_levels_request, pressure_file)
        return pressure_file

    def fetch_single_levels_data(self, year, month, day, hour, latitude, longitude):
        
        single_levels_request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_temperature',
                '2m_relative_humidity',
                'mean_sea_level_pressure',
                'surface_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'total_column_water_vapour',
                'total_precipitation',
            ],
            'year': str(year),
            'month': str(month).zfill(2),
            'day': str(day).zfill(2),
            'time': f'{str(hour).zfill(2)}:00',
            'area': [
                latitude + 0.25, longitude - 0.25,
                latitude - 0.25, longitude + 0.25,
            ],
        }
        
        # Download single levels data
        single_file = f'era5_single_levels_{year}{month:02d}{day:02d}_{hour:02d}.nc'
        self.client.retrieve('reanalysis-era5-single-levels', single_levels_request, single_file)
        return single_file

    def process_data(self, pressure_file, single_file):
        
        # Read NetCDF files
        ds_pressure = xr.open_dataset(pressure_file)
        ds_single = xr.open_dataset(single_file)
        
        # Convert geopotential to geopotential height (divide by g = 9.80665)
        g = 9.80665
        
        # Initialize dictionary for DataFrame
        data = {}
        
        # Process single level variables
        data['temperature_2m'] = ds_single['t2m'].values[0, 0, 0] - 273.15  # Convert K to °C
        data['relative_humidity_2m'] = ds_single['r2'].values[0, 0, 0]
        data['pressure_msl'] = ds_single['msl'].values[0, 0, 0] / 100  # Convert Pa to hPa
        data['surface_pressure'] = ds_single['sp'].values[0, 0, 0] / 100  # Convert Pa to hPa
        
        # Calculate 10m wind speed and direction from components
        u10 = ds_single['u10'].values[0, 0, 0]
        v10 = ds_single['v10'].values[0, 0, 0]
        data['wind_speed_10m'] = np.sqrt(u10**2 + v10**2)
        data['wind_direction_10m'] = np.mod(180 + np.degrees(np.arctan2(u10, v10)), 360)
        
        # Process pressure level variables
        for level in ['1000', '850', '500', '50']:
            idx = ds_pressure.level == float(level)
            
            if level != '50':  # All variables except for 50hPa
                data[f'temperature_{level}hPa'] = ds_pressure['t'].values[0, idx, 0, 0][0] - 273.15
                
                u = ds_pressure['u'].values[0, idx, 0, 0][0]
                v = ds_pressure['v'].values[0, idx, 0, 0][0]
                data[f'wind_speed_{level}hPa'] = np.sqrt(u**2 + v**2)
                data[f'wind_direction_{level}hPa'] = np.mod(180 + np.degrees(np.arctan2(u, v)), 360)
                
                if level in ['850', '500']:  # RH only for 850 and 500
                    data[f'relative_humidity_{level}hPa'] = ds_pressure['r'].values[0, idx, 0, 0][0]
            
            # Geopotential height for all levels
            data[f'geopotential_height_{level}hPa'] = ds_pressure['z'].values[0, idx, 0, 0][0] / g
        
        # Total column water vapor
        data['total_column_integrated_water_vapour'] = ds_single['tcwv'].values[0, 0, 0]
        
        # Clean up
        ds_pressure.close()
        ds_single.close()
        os.remove(pressure_file)
        os.remove(single_file)
        
        return pd.DataFrame([data])

def fetch_era5_data(latitude, longitude, timezone):
    
    # Initialize ERA5 fetcher
    fetcher = ERA5Fetcher()
    
    # Get current time in UTC
    current_time = pd.Timestamp.utcnow()
    
    # ERA5 data has about 5 days delay, so we fetch data from 6 days ago
    target_time = current_time - pd.Timedelta(days=6)
    
    # Fetch data
    pressure_file = fetcher.fetch_pressure_levels_data(
        target_time.year, target_time.month, target_time.day, 
        target_time.hour, latitude, longitude
    )
    
    single_file = fetcher.fetch_single_levels_data(
        target_time.year, target_time.month, target_time.day,
        target_time.hour, latitude, longitude
    )
    
    # Process data
    return fetcher.process_data(pressure_file, single_file)"""