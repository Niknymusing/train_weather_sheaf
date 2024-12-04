import streamlit as st
import socket
import json
import pandas as pd
import plotly.express as px
import threading

def send_weather_request(latitude, longitude, timezone="UTC", port=5001):
    """Send a weather prediction request to the local server."""
    # Create request data
    request = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone
    }
    
    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect(('localhost', port))
        
        # Send request
        client_socket.send(json.dumps(request).encode())
        
        # Receive response
        response = ""
        while True:
            data = client_socket.recv(4096).decode()
            if not data:
                break
            response += data

        if response:
            # Parse and return response
            result = json.loads(response)
            return result
        else:
            st.error("No response received from server.")
            return None

    except ConnectionRefusedError:
        st.error(f"Could not connect to server on port {port}. Is the server running?")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        st.error(f"Traceback:\n{traceback_str}")
        return None
    finally:
        client_socket.close()

def send_weather_request_thread(latitude, longitude, result_container, timezone="UTC", port=5001):
    result_container['data'] = send_weather_request(latitude, longitude, timezone, port)

# Set the app title and description
st.set_page_config(page_title="Weather AI Query App", layout="wide")
st.title("🌤️ Weather AI Query App")
st.write("""
Enter the latitude and longitude coordinates below and click on **Query Weather AI** to get the latest weather predictions, probability distributions, and key metrics.
""")

# Create input fields
col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input("Latitude", value=59.3293, format="%f")
with col2:
    longitude = st.number_input("Longitude", value=18.0686, format="%f")

# Add a button to trigger the query
if st.button("Query Weather AI"):
    try:
        result_container = {}
        with st.spinner("Querying Weather AI..."):
            thread = threading.Thread(target=send_weather_request_thread, args=(latitude, longitude, result_container))
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
                    metric_cols[idx].metric(label=metric_name.capitalize(), value=f"{value:.4f}")
            else:
                st.write("No metrics data available.")
            
            # Display Weather Data
            st.header("🌡️ Weather Data")
            weather_data = result.get('weather_data', [])
            if weather_data:
                df_weather = pd.DataFrame(weather_data)
                df_weather['date'] = pd.to_datetime(df_weather['date'])
                st.dataframe(df_weather)
                
                # Plot temperature over time
                fig_temp = px.line(df_weather, x='date', y='T2m', title='Temperature Over Time', labels={'T2m': 'Temperature (°C)'})
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.write("No weather data available.")
            
            # Display Probability Distribution
            st.header("📈 Probability Distribution")
            model_output = result.get('model_output', [])
            if model_output:
                probabilities = model_output[0]
                df_prob = pd.DataFrame(probabilities, columns=['Probability'])
                df_prob['Index'] = df_prob.index
                fig_prob = px.bar(df_prob, x='Index', y='Probability', title='Probability Distribution')
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.write("No model output data available.")
        else:
            st.error("Failed to retrieve data.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        st.error(f"Traceback:\n{traceback_str}")

