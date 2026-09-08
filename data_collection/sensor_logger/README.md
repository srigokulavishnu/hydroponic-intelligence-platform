# Raspberry Pi Dataset Collection Module

## 1. Project Purpose
This module is a Hydroponic Intelligence Platform component designed for a Raspberry Pi based dataset collection system growing Palak (Spinach). It is built to strictly and passively monitor an existing hydroponic automation system without interfering with the main control system. The dataset collector's purpose is to read sensor data, store research dataset files locally, and synchronize sensor records and images.

## 2. Architecture
The dataset collector runs on a separate Raspberry Pi connected to the same LAN as the main hydroponic server. It fetches data periodically and logs it directly to a CSV file.

```text
Existing Sensor API
        │
        ▼
sensor_logger.py
        │
        ▼
sensor_environment.csv
```

Later, a camera collection pipeline will be integrated alongside this logger for multimodal dataset generation.

## 3. API URL
- **API URL:** `http://192.168.100.131:5000/api/data`
- The script uses the local network to hit this endpoint and parse the results via a GET request.

## 4. Installation

Ensure you have Python 3 installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## 5. How to Run

Execute the Python application:

```bash
python sensor_logger.py
```

## 6. Expected CSV Location
- **Data CSV Location:** `data/sensor_environment.csv` (Created automatically if it doesn't exist)

## 7. Expected Log Location
- **Log File Location:** `logs/sensor_logger.log` (Created automatically if it doesn't exist)

## 8. Network Failures Handling
The program is built with high robustness. It handles LAN disconnections, server restarts, dashboard restarts, API timeouts, and Raspberry Pi starting before the dashboard server gracefully. 
The script catches request exceptions and logs errors appropriately. Instead of crashing, it will wait for the next interval to try again and will automatically reconnect when the server becomes available again.

## 9. Important Safety Note
This dataset collector is **PASSIVE and READ-ONLY**.
It only performs `GET` requests. It will **NEVER**:
- POST to the existing hydroponic server
- PUT data or DELETE data
- Change relays, pumps, or control fans/foggers
- Change Modbus settings
- Modify the dashboard database or the main control program
- Send actuator commands
