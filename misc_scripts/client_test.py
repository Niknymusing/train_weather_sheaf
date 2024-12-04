import socket
import json
import sys

def send_weather_request(latitude, longitude, timezone="UTC", port=5001):
    """Send a weather prediction request to the local server."""
    
    # Create request data
    request = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone
    }
    
    print(f"Connecting to server on port {port}...")
    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect(('localhost', port))
        print("Connected to server")
        
        # Send request
        print(f"Sending request: {request}")
        client_socket.send(json.dumps(request).encode())
        print("Request sent")
        
        # Receive response
        print("Waiting for response...")
        response = ""
        while True:
            data = client_socket.recv(4096).decode()
            print(f"Received chunk of size: {len(data) if data else 0}")
            if not data:
                break
            response += data
            
        if response:
            # Parse and print response
            print("Parsing response...")
            result = json.loads(response)
            
            print("\nReceived prediction results:")
            print("---------------------------")
            
            # Print weather data summary
            weather_data = result['weather_data']
            if weather_data:
                print("\nWeather Data:")
                print(f"Number of timestamps: {len(weather_data)}")
                print("Available variables:", list(weather_data[0].keys()))
                
                # Print a sample of temperature data
                print("\nSample temperature readings:")
                for entry in weather_data[:5]:  # First 5 entries
                    print(f"Time: {entry['date']}, Temperature: {entry['T2m']}°C")
            
            # Print model output summary
            print("\nModel Output Summary:")
            model_output = result['model_output']
            print(f"Output shape: {len(model_output)} values")
            print("First 5 values:", model_output[:5])
            
            # Print metrics
            print("\nMetrics:")
            for metric_name, value in result['metrics'].items():
                print(f"{metric_name}: {value}")
            
        else:
            print("No response received from server")
        
    except ConnectionRefusedError:
        print(f"Could not connect to server on port {port}. Is the server running?")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing connection")
        client_socket.close()

def main():
    # Get coordinates from command line or use defaults
    if len(sys.argv) >= 3:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
    else:
        print("Please enter coordinates:")
        try:
            lat = float(input("Latitude (e.g. 59.3293 for Stockholm): "))
            lon = float(input("Longitude (e.g. 18.0686 for Stockholm): "))
        except ValueError:
            print("Invalid coordinates. Using Stockholm coordinates as default.")
            lat, lon = 59.3293, 18.0686

    print(f"\nSending request for coordinates: {lat}, {lon}")
    send_weather_request(lat, lon)

if __name__ == "__main__":
    main()