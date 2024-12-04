import streamlit as st
import socket
import json
import pandas as pd
import plotly.express as px
import threading
import numpy as np
from scipy import stats
import io

def generate_reference_distribution(dist_type, size=1000):
    if dist_type == "normal":
        mu = np.random.uniform(-2, 2)
        sigma = np.random.uniform(0.5, 2)
        return stats.norm.pdf(np.linspace(-5, 5, size), mu, sigma)
    elif dist_type == "exponential":
        scale = np.random.uniform(0.5, 2)
        return stats.expon.pdf(np.linspace(0, 10, size), scale=scale)
    elif dist_type == "student_t":
        df = np.random.randint(2, 10)
        return stats.t.pdf(np.linspace(-5, 5, size), df)
    

def send_weather_request(latitude, longitude, reference_dist=None, timezone="UTC", port=5001):
    request = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "reference_distribution": reference_dist.tolist() if reference_dist is not None else None
    }
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect(('localhost', port))
        
        # Send data length first, then data
        data = json.dumps(request).encode()
        data_length = len(data)
        client_socket.send(str(data_length).encode() + b'\n')
        client_socket.sendall(data)
        
        # Receive response length, then data
        response_length = int(client_socket.recv(1024).decode().strip())
        response = b""
        while len(response) < response_length:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            response += chunk
            
        if response:
            return json.loads(response.decode())
        return None
    finally:
        client_socket.close()



def send_weather_request_thread(latitude, longitude, result_container, reference_dist=None, timezone="UTC", port=5001):
    result_container['data'] = send_weather_request(latitude, longitude, reference_dist, timezone, port)

def validate_uploaded_distribution(data):
    try:
        dist = np.array(data)
        if len(dist) != 1000:
            return None, "Distribution must contain exactly 1000 values"
        if np.any(dist < 0):
            return None, "Distribution cannot contain negative values"
        # Normalize if not already normalized
        dist = dist / np.sum(dist)
        return dist, None
    except Exception as e:
        return None, f"Invalid distribution format: {str(e)}"

st.set_page_config(page_title="Weather AI Query App", layout="wide")
st.title("🌤️ Weather AI Query App")
st.write("Enter coordinates and reference distribution settings below.")

# Input coordinates
col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input("Latitude", value=59.3293, format="%f")
with col2:
    longitude = st.number_input("Longitude", value=18.0686, format="%f")

# Reference distribution section
st.header("📊 Reference Distribution")
dist_method = st.radio("Choose reference distribution method:", 
                      ["Generate Random", "Upload Custom"])

reference_dist = None
if dist_method == "Generate Random":
    dist_type = st.selectbox("Distribution Type:", 
                            ["normal", "exponential", "student_t"])
    if st.button("Generate Distribution"):
        reference_dist = generate_reference_distribution(dist_type)
        fig_ref = px.line(x=range(1000), y=reference_dist, 
                         title=f"Generated {dist_type.capitalize()} Distribution")
        st.plotly_chart(fig_ref, use_container_width=True)
else:
    uploaded_file = st.file_uploader("Upload distribution file (CSV or TXT)", 
                                   type=['csv', 'txt'])
    if uploaded_file:
        try:
            content = pd.read_csv(uploaded_file, header=None).values.flatten()
            reference_dist, error_msg = validate_uploaded_distribution(content)
            if error_msg:
                st.error(error_msg)
            else:
                fig_ref = px.line(x=range(1000), y=reference_dist, 
                                title="Uploaded Distribution")
                st.plotly_chart(fig_ref, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Query button and results
if st.button("Query Weather AI"):
    try:
        result_container = {}
        with st.spinner("Querying Weather AI..."):
            thread = threading.Thread(
                target=send_weather_request_thread,
                args=(latitude, longitude, result_container, reference_dist)
            )
            thread.start()
            thread.join()

        result = result_container.get('data')
        if result:
            st.success("Data received successfully!")
            
            # Display Metrics
            st.header("📊 Metrics")
            metrics = result.get('metrics', {})
            if metrics:
                metric_cols = st.columns(len(metrics))
                for idx, (metric_name, value) in enumerate(metrics.items()):
                    metric_cols[idx].metric(label=metric_name.capitalize(), 
                                         value=f"{value:.4f}")
            
            # Display Weather Data
            st.header("🌡️ Weather Data")
            weather_data = result.get('weather_data', [])
            if weather_data:
                df_weather = pd.DataFrame(weather_data)
                df_weather['date'] = pd.to_datetime(df_weather['date'])
                st.dataframe(df_weather)
                
                # Temperature at 2 meters
                fig_temp = px.line(df_weather, x='date', y='T2m', 
                                title='Temperature Over Time',
                                labels={'T2m': 'Temperature (°C)'})
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Surface Pressure
                fig_temp = px.line(df_weather, x='date', y='sp', 
                                title='Surface Pressure',
                                labels={'sp': 'Surface Pressure (hPa)'})
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Mean Sea Level Pressure
                fig_temp = px.line(df_weather, x='date', y='mslp', 
                                title='Mean Sea Level Pressure',
                                labels={'mslp': 'Mean Sea Level Pressure (hPa)'})
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Geopotential Height at 1000 hPa
                fig_temp = px.line(df_weather, x='date', y='Z_1000hPa', 
                                title='Geopotential Height at 1000 hPa',
                                labels={'Z_1000hPa': 'Geopotential Height (m)'})
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Temperature at 850 hPa
                fig_temp = px.line(df_weather, x='date', y='T_850hPa', 
                                title='Temperature at 850 hPa',
                                labels={'T_850hPa': 'Temperature (°C)'})
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Geopotential Height at 850 hPa
                fig_temp = px.line(df_weather, x='date', y='Z_850hPa', 
                                title='Geopotential Height at 850 hPa',
                                labels={'Z_850hPa': 'Geopotential Height (m)'})
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Relative Humidity at 850 hPa
                fig_temp = px.line(df_weather, x='date', y='RH_850hPa', 
                                title='Relative Humidity at 850 hPa',
                                labels={'RH_850hPa': 'Relative Humidity (%)'})
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Temperature at 500 hPa
                fig_temp = px.line(df_weather, x='date', y='T_500hPa', 
                                title='Temperature at 500 hPa',
                                labels={'T_500hPa': 'Temperature (°C)'})
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Geopotential Height at 500 hPa
                fig_temp = px.line(df_weather, x='date', y='Z_500hPa', 
                                title='Geopotential Height at 500 hPa',
                                labels={'Z_500hPa': 'Geopotential Height (m)'})
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Relative Humidity at 500 hPa
                fig_temp = px.line(df_weather, x='date', y='RH_500hPa', 
                                title='Relative Humidity at 500 hPa',
                                labels={'RH_500hPa': 'Relative Humidity (%)'})
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Geopotential Height at 50 hPa
                fig_temp = px.line(df_weather, x='date', y='Z_50hPa', 
                                title='Geopotential Height at 50 hPa',
                                labels={'Z_50hPa': 'Geopotential Height (m)'})
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Total Column Integrated Water Vapour
                fig_temp = px.line(df_weather, x='date', y='T CW V', 
                                title='Total Column Integrated Water Vapour',
                                labels={'T CW V': 'Water Vapour (kg/m²)'})
                st.plotly_chart(fig_temp, use_container_width=True)

            
            # Display Model Output Distribution
            st.header("📈 Model Output Distribution")
            model_output = result.get('model_output', [])
            if model_output:
                #st.write("Received model_output:", model_output)
                probabilities = model_output  # Removed [0] index
                df_prob = pd.DataFrame({'Probability': probabilities})
                df_prob['Index'] = df_prob.index
                fig_prob = px.line(df_prob, x='Index', y='Probability', 
                                title='Model Output Distribution')
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.error("No model_output received.")

        else:
            st.error("Weather data fetch failed. Try again by Query Weather AI!")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
