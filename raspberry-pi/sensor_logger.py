import time
import json
import csv
import logging
from datetime import datetime
import requests
from requests.exceptions import RequestException
import config

def setup_directories():
    """Ensure data and logs directories exist."""
    config.DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

def setup_logging():
    """Configure logging to file and console."""
    log_file = config.LOG_DIRECTORY / config.LOG_FILENAME
    
    # Reset existing handlers to avoid duplicates if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def fetch_sensor_data():
    """Fetch data from the sensor API."""
    try:
        response = requests.get(config.API_URL, timeout=config.REQUEST_TIMEOUT)
        return response
    except RequestException as e:
        logging.error(f"API connection failed: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during API request: {e}")
        return None

def validate_api_response(response):
    """Validate the HTTP response and JSON structure."""
    if response is None:
        return None
    
    if response.status_code != 200:
        logging.error(f"API returned non-200 status code: {response.status_code}")
        return None
        
    try:
        data = response.json()
    except ValueError:
        logging.error("API response is not valid JSON")
        return None
        
    if data.get("status") != "ok":
        logging.error(f"API returned status '{data.get('status')}' instead of 'ok'")
        return None
        
    if "data" not in data:
        logging.error("API response JSON does not contain 'data' object")
        return None
        
    return data["data"]

def extract_sensor_record(sensor_data):
    """Extract and format the specific fields needed for the dataset."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract devices explicitly
    devices = sensor_data.get("devices", {})
    
    # Mapping to handle missing data gracefully
    def get_val(source, key):
        val = source.get(key)
        if val is None:
            logging.warning(f"Missing field: {key}")
            return ""
        return val

    record = {
        "timestamp": timestamp,
        "device_online": get_val(sensor_data, "device_online"),
        "ph": get_val(sensor_data, "ph"),
        "ec": get_val(sensor_data, "ec"),
        "tds": get_val(sensor_data, "tds"),
        "salinity": get_val(sensor_data, "salinity"),
        "temperature1": get_val(sensor_data, "temperature1"),
        "temperature2": get_val(sensor_data, "temperature2"),
        "temperature_avg": get_val(sensor_data, "temp_avg"),
        "humidity1": get_val(sensor_data, "humidity1"),
        "humidity2": get_val(sensor_data, "humidity2"),
        "humidity_avg": get_val(sensor_data, "humidity_avg"),
        "ldr1": get_val(sensor_data, "ldr1"),
        "ldr2": get_val(sensor_data, "ldr2"),
        "fan1": get_val(devices, "fan1"),
        "fan2": get_val(devices, "fan2"),
        "fogger": get_val(devices, "fogger"),
        "honeycomb": get_val(devices, "honeycomb"),
        "pump1": get_val(devices, "pump1"),
        "pump2": get_val(devices, "pump2"),
        "pump3": get_val(devices, "pump3"),
        "pump4": get_val(devices, "pump4")
    }
    return record

def initialize_csv(csv_path, fieldnames):
    """Create CSV and write headers if it doesn't exist."""
    if not csv_path.exists():
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            logging.info("Created new CSV file and wrote headers")
        except Exception as e:
            logging.error(f"Failed to initialize CSV: {e}")

def append_to_csv(csv_path, record, fieldnames):
    """Append a single record to the CSV file."""
    try:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(record)
            f.flush()
        return True
    except Exception as e:
        logging.error(f"Failed to append to CSV: {e}")
        return False

def print_sensor_data(record):
    """Display the collected data nicely in the terminal."""
    print("\n" + "="*48)
    print("Hydroponic Sensor Dataset Logger".center(48))
    print("="*48 + "\n")
    
    print(f"Timestamp: {record['timestamp']}\n")
    print(f"Device Online: {record['device_online']}\n")
    
    print(f"pH: {record['ph']}")
    print(f"EC: {record['ec']} µS/cm" if record['ec'] != "" else "EC: ")
    print(f"TDS: {record['tds']} PPM" if record['tds'] != "" else "TDS: ")
    print(f"Salinity: {record['salinity']} PPT\n" if record['salinity'] != "" else "Salinity: \n")
    
    print(f"Temperature 1: {record['temperature1']} °C" if record['temperature1'] != "" else "Temperature 1: ")
    print(f"Temperature 2: {record['temperature2']} °C" if record['temperature2'] != "" else "Temperature 2: ")
    print(f"Temperature Average: {record['temperature_avg']} °C\n" if record['temperature_avg'] != "" else "Temperature Average: \n")
    
    print(f"Humidity 1: {record['humidity1']} %" if record['humidity1'] != "" else "Humidity 1: ")
    print(f"Humidity 2: {record['humidity2']} %" if record['humidity2'] != "" else "Humidity 2: ")
    print(f"Humidity Average: {record['humidity_avg']} %\n" if record['humidity_avg'] != "" else "Humidity Average: \n")
    
    print(f"LDR1: {record['ldr1']} lux" if record['ldr1'] != "" else "LDR1: ")
    print(f"LDR2: {record['ldr2']} lux\n" if record['ldr2'] != "" else "LDR2: \n")
    
    print(f"Fan1: {record['fan1']}")
    print(f"Fan2: {record['fan2']}")
    print(f"Fogger: {record['fogger']}")
    print(f"Honeycomb: {record['honeycomb']}\n")
    
    print(f"Pump1: {record['pump1']}")
    print(f"Pump2: {record['pump2']}")
    print(f"Pump3: {record['pump3']}")
    print(f"Pump4: {record['pump4']}\n")
    
    print("Sensor data saved successfully.")
    print("="*48 + "\n")

def run_logger():
    """Main execution loop for the logger."""
    setup_directories()
    setup_logging()
    
    logging.info("Application startup. Initializing sensor logger...")
    
    csv_path = config.DATA_DIRECTORY / config.CSV_FILENAME
    fieldnames = [
        "timestamp", "device_online", "ph", "ec", "tds", "salinity",
        "temperature1", "temperature2", "temperature_avg",
        "humidity1", "humidity2", "humidity_avg",
        "ldr1", "ldr2",
        "fan1", "fan2", "fogger", "honeycomb",
        "pump1", "pump2", "pump3", "pump4"
    ]
    
    initialize_csv(csv_path, fieldnames)
    logging.info("Sensor logger initialized and ready.")
    
    while True:
        try:
            response = fetch_sensor_data()
            sensor_data = validate_api_response(response)
            
            if sensor_data:
                record = extract_sensor_record(sensor_data)
                
                if append_to_csv(csv_path, record, fieldnames):
                    logging.info("Sensor data collected successfully")
                    print_sensor_data(record)
            
        except Exception as e:
            logging.error(f"Unexpected exception in main loop: {e}")
            
        # Wait until next collection interval
        time.sleep(config.COLLECTION_INTERVAL)

if __name__ == "__main__":
    run_logger()
