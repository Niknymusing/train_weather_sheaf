from snet import sdk
import phot_config
import numpy as np
#import adr_pb2
#import adr_pb2_grpc
from requests.exceptions import HTTPError
import time
import base64
import io
import torch
import logging
import datetime
from photrek_algo import compute_metrics


class SnetConfig(dict):
    def __init__(self):
        if not hasattr(phot_config, 'PRIVATE_KEY') or not hasattr(phot_config, 'ETH_RPC_ENDPOINT'):
            raise ValueError("Missing PRIVATE_KEY or ETH_RPC_ENDPOINT in config")
        
        super().__init__({
            "private_key": phot_config.PRIVATE_KEY,
            "eth_rpc_endpoint": phot_config.ETH_RPC_ENDPOINT,
            "identity_name": "prod_user",
            "identity_type": "key",
            "network": "mainnet",
            "lighthouse_token": "f2548d27ffd319b9c05918eeac15ebab934e5cfcd68e1ec3db2b92765",
            "request_timeout": 60,
        })
        
    def get_ipfs_endpoint(self):
        return "/dns4/ipfs.singularitynet.io/tcp/80/http"

def create_test_data(num_rows=1000, num_classes=2):
    
    # Create string buffer to write CSV data
    buffer = io.StringIO()
    
    # Generate data following the makeExample.py format
    for _ in range(num_rows):
        # Generate random probabilities
        probs = [str(np.random.rand()) for _ in range(num_classes)]
        # Generate random class (1-indexed)
        class_label = str(np.random.randint(num_classes) + 1)
        # Write row
        buffer.write(','.join(probs + [class_label]) + '\n')
    
    # Get the string content
    csv_content = buffer.getvalue()
    
    # Encode as base64
    csv_base64 = base64.b64encode(csv_content.encode()).decode()
    
    # Create input string in format: numRows,numCols,base64EncodedCSV
    input_string = f"{num_rows},{num_classes+1},{csv_base64}"  # num_classes+1 because class column is included
    
    return input_string

def retry_with_delay(func, max_retries=5, delay=5):
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting {func.__name__}... (attempt {attempt + 1}/{max_retries})")
            return func()
        except HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    sleep_time = delay * (2 ** attempt)
                    print(f"Rate limit hit. Waiting {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            print(f"Unexpected error in {func.__name__}: {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = delay * (2 ** attempt)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            raise


def call_photrek(p, q):
    """
    Call the Photrek service to compute the three comparison metrics between two probability distributions.
    Returns the same format as compute_metrics function.
    
    Args:
        p: First probability distribution (tensor or array-like)
        q: Second probability distribution (tensor or array-like)
        
    Returns:
        dict: Dictionary with keys 'accuracy', 'decisiveness', 'robustness'
    """
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'photrek_api_{timestamp}.log'
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Log input distributions
    logging.info("=== Starting Photrek API Call ===")
    logging.info(f"Input p type: {type(p)}, shape: {p.shape if hasattr(p, 'shape') else 'scalar'}")
    logging.info(f"Input q type: {type(q)}, shape: {q.shape if hasattr(q, 'shape') else 'scalar'}")
    
    # Ensure inputs are tensors
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    else:
        p = p.clone().detach()
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32)
    else:
        q = q.clone().detach()
    
    # Log tensor conversions
    logging.debug(f"Converted p tensor: {p}")
    logging.debug(f"Converted q tensor: {q}")
    
    # Ensure numerical stability
    eps = 1e-10
    p = torch.clamp(p, min=eps)
    p = p / p.sum()
    q = torch.clamp(q, min=eps)
    q = q / q.sum()
    
    # Log normalized distributions
    logging.debug(f"Normalized p: {p}")
    logging.debug(f"Normalized q: {q}")
    
    try:
        # Create configuration
        config_obj = SnetConfig()
        logging.info("SnetConfig created successfully")
        
        # Initialize SDK
        def init_sdk():
            return sdk.SnetSDK(config_obj)
        snet_sdk = retry_with_delay(init_sdk)
        logging.info("SDK initialized successfully")
        
        # Create service client
        def create_client():
            return snet_sdk.create_service_client(
                org_id="Photrek",
                service_id="risk-aware-assessment",
                group_name="default_group"
            )
        service_client = retry_with_delay(create_client)
        logging.info("Service client created successfully")
        
        # Create string buffer to write CSV data
        buffer = io.StringIO()
        
        # Convert tensors to numpy for string formatting
        p_np = p.numpy()
        q_np = q.numpy()
        num_classes = len(p_np)
        
        # Log numpy conversions
        logging.debug(f"Numpy p: {p_np}")
        logging.debug(f"Numpy q: {q_np}")
        logging.info(f"Number of classes: {num_classes}")
        
        # Write first distribution as row with class label 1
        probs = [str(x) for x in p_np]
        buffer.write(','.join(probs + ['1']) + '\n')
            
        # Write second distribution as row with class label 2
        probs = [str(x) for x in q_np]
        buffer.write(','.join(probs + ['2']) + '\n')
        
        # Get the string content and log it
        csv_content = buffer.getvalue()
        logging.debug(f"Generated CSV content:\n{csv_content}")
        
        # Encode as base64
        csv_base64 = base64.b64encode(csv_content.encode()).decode()
        logging.debug(f"Base64 encoded length: {len(csv_base64)}")
        
        # Create input string
        input_str = f"2,{num_classes+1},{csv_base64}"
        logging.info(f"Input string format: rows=2, columns={num_classes+1}")
        
        def call_service():
            logging.info("Calling Photrek API...")
            response = service_client.call_rpc(
                rpc_name="adr",
                message_class="InputString",
                s=input_str
            )
            
            # Log detailed response information
            logging.info("=== API Response Details ===")
            logging.info(f"Response type: {type(response)}")
            logging.info(f"Available attributes: {dir(response)}")
            
            # Log all non-private attributes and their types
            for attr in dir(response):
                if not attr.startswith('_'):
                    try:
                        value = getattr(response, attr)
                        logging.info(f"Attribute '{attr}': {type(value)} = {value}")
                    except Exception as e:
                        logging.error(f"Error accessing attribute {attr}: {str(e)}")
            
            return response
            
        response = retry_with_delay(call_service)
        
        if hasattr(response, 'a'):
            metrics = {
                "accuracy": response.a,
                "decisiveness": response.d,
                "robustness": response.r
            }
            logging.info(f"Computed metrics: {metrics}")
            return metrics
        else:
            error_msg = "Unexpected response format from service"
            logging.error(f"{error_msg}\nResponse attributes: {dir(response)}")
            raise Exception(error_msg)
            
    except Exception as e:
        logging.error(f"Error in call_photrek: {str(e)}", exc_info=True)
        raise

def __call_photrek(p, q):
    """
    Call the Photrek service to compute the three comparison metrics between two probability distributions.
    Returns the same format as compute_metrics function.
    
    Args:
        p: First probability distribution (tensor or array-like)
        q: Second probability distribution (tensor or array-like)
        
    Returns:
        dict: Dictionary with keys 'accuracy', 'decisiveness', 'robustness'
    """
    # Ensure inputs are tensors
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    else:
        p = p.clone().detach()
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32)
    else:
        q = q.clone().detach()
    
    # Ensure numerical stability
    eps = 1e-10
    p = torch.clamp(p, min=eps)
    p = p / p.sum()
    q = torch.clamp(q, min=eps)
    q = q / q.sum()
    
    try:
        # Create configuration
        config_obj = SnetConfig()
        
        # Initialize SDK
        def init_sdk():
            return sdk.SnetSDK(config_obj)
        snet_sdk = retry_with_delay(init_sdk)
        
        # Create service client
        def create_client():
            return snet_sdk.create_service_client(
                org_id="Photrek",
                service_id="risk-aware-assessment",
                group_name="default_group"
            )
        service_client = retry_with_delay(create_client)
        
        # Create string buffer to write CSV data
        buffer = io.StringIO()
        
        # Convert tensors to numpy for string formatting
        p_np = p.numpy()
        q_np = q.numpy()
        num_classes = len(p_np)
        
        # Write first distribution as row with class label 1
        probs = [str(x) for x in p_np]
        buffer.write(','.join(probs + ['1']) + '\n')
            
        # Write second distribution as row with class label 2
        probs = [str(x) for x in q_np]
        buffer.write(','.join(probs + ['2']) + '\n')
        
        # Get the string content
        csv_content = buffer.getvalue()
        
        # Encode as base64
        csv_base64 = base64.b64encode(csv_content.encode()).decode()
        
        # Create input string - only 2 rows, each row has num_classes + 1 columns
        input_str = f"2,{num_classes+1},{csv_base64}"
        
        def call_service():
            return service_client.call_rpc(
                rpc_name="adr",
                message_class="InputString",
                s=input_str
            )
            
        response = retry_with_delay(call_service)
        
        if hasattr(response, 'a'):
            return {
                "accuracy": response.a,
                "decisiveness": response.d,
                "robustness": response.r
            }
        else:
            raise Exception("Unexpected response format from service")
            
    except Exception as e:
        print(f"Error in call_photrek: {str(e)}")
        raise


def _call_photrek(p, q):
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



def main():
    try:
        # Create configuration
        config_obj = SnetConfig()
        print("Configuration created")
        
        # Initialize SDK
        def init_sdk():
            return sdk.SnetSDK(config_obj)
        snet_sdk = retry_with_delay(init_sdk)
        print("SDK initialized successfully")
        
        # Create service client
        def create_client():
            return snet_sdk.create_service_client(
                org_id="Photrek",
                service_id="risk-aware-assessment",
                group_name="default_group"
            )
        service_client = retry_with_delay(create_client)
        print("Service client created successfully")
        
        # Create test data exactly as in makeExample.py
        input_str = create_test_data(num_rows=1000, num_classes=2)
        print(f"\nCreated test data with 1000 rows and 2 classes")
        
        def call_service():
            print("Calling ADR service...")
            return service_client.call_rpc(
                rpc_name="adr",
                message_class="InputString",
                s=input_str
            )
            
        response = retry_with_delay(call_service)
        
        # Process and display results
        print("\nService Response:")
        if hasattr(response, 'a'):
            print(f"Alignment (A): {response.a:.4f}")
            print(f"Distinctness (D): {response.d:.4f}")
            print(f"Robustness (R): {response.r:.4f}")
            
            if hasattr(response, 'img') and response.img:
                print(f"\nReceived image data ({response.numr}x{response.numc})")
                try:
                    with open("output_plot.png", 'wb') as f:
                        f.write(base64.b64decode(response.img))
                    print("Plot saved as output_plot.png")
                except Exception as e:
                    print(f"Failed to save plot: {str(e)}")
        else:
            print("Unexpected response format:", response)
        
        return response

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()




def main():
    logger = setup_logging()
    try:
        # Create configuration
        config_obj = SnetConfig()
        logger.info("Configuration created")
        
        # Initialize SDK
        def init_sdk():
            return sdk.SnetSDK(config_obj)
        snet_sdk = retry_with_delay(init_sdk)
        logger.info("SDK initialized successfully")
        
        # Create service client
        def create_client():
            return snet_sdk.create_service_client(
                org_id="Photrek",
                service_id="risk-aware-assessment",
                group_name="default_group"
            )
        service_client = retry_with_delay(create_client)
        logger.info("Service client created successfully")
        
        # Create test data
        input_str = create_test_data(num_rows=1000, num_classes=2)
        logger.info(f"Created test data with 1000 rows and 4 classes")
        
        def call_service():
            logger.info("Calling ADR service...")
            response = service_client.call_rpc(
                rpc_name="adr",
                message_class="InputString",
                s=input_str
            )
            
            # Log detailed response information
            logger.info("=== API Response Details ===")
            logger.info(f"Response type: {type(response)}")
            
            # Log all attributes and their values
            for attr in dir(response):
                if not attr.startswith('_'):  # Skip private attributes
                    try:
                        value = getattr(response, attr)
                        if isinstance(value, bytes):
                            logger.info(f"Attribute '{attr}': {type(value)} (length: {len(value)})")
                        else:
                            logger.info(f"Attribute '{attr}': {type(value)} = {value}")
                    except Exception as e:
                        logger.error(f"Error accessing attribute {attr}: {str(e)}")
            
            return response
            
        response = retry_with_delay(call_service)
        
        # Process and display results
        logger.info("\nService Response:")
        if hasattr(response, 'a'):
            logger.info(f"Alignment (A): {response.a:.4f}")
            logger.info(f"Distinctness (D): {response.d:.4f}")
            logger.info(f"Robustness (R): {response.r:.4f}")
            
            if hasattr(response, 'img') and response.img:
                logger.info(f"Received image data ({response.numr}x{response.numc})")
                try:
                    with open("output_plot.png", 'wb') as f:
                        f.write(base64.b64decode(response.img))
                    logger.info("Plot saved as output_plot.png")
                except Exception as e:
                    logger.error(f"Failed to save plot: {str(e)}")
        else:
            logger.error("Unexpected response format:", response)
        
        return response

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()

