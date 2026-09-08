import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# ---------------------------------------------------------
# CONSTANTS & SETUP
# ---------------------------------------------------------

# Change working directory to the script's directory so relative paths always work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

CONFIG_FILE = "config.json"

def load_config(config_path=CONFIG_FILE):
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(log_file):
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

# ---------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------

def load_state(state_file):
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Corrupted state file {state_file}. Starting fresh.")
    
    return {
        "processed_captures": []
    }

def save_state(state, state_file):
    state_dir = os.path.dirname(state_file)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)
        
    # Write to temporary file then rename to prevent corruption
    tmp_file = state_file + ".tmp"
    with open(tmp_file, 'w') as f:
        json.dump(state, f, indent=4)
    os.replace(tmp_file, state_file)

# ---------------------------------------------------------
# SENSOR PROCESSING
# ---------------------------------------------------------

def update_sensor_master_dataset(config):
    sensor_csv = config['sensor_csv']
    sensor_master = config['sensor_master_dataset']
    
    if not os.path.exists(sensor_csv):
        logging.warning(f"Sensor CSV not found at {sensor_csv}")
        return None
        
    try:
        # We read the entire raw CSV. Since V1 correctness > optimization, this is fine.
        df = pd.read_csv(sensor_csv)
        
        # Ensure timestamp is parsed correctly
        if 'timestamp' in df.columns:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        else:
            logging.error("No 'timestamp' column in sensor CSV.")
            return None
            
        # Ensure the sensor master dataset directory exists
        os.makedirs(os.path.dirname(sensor_master), exist_ok=True)
        
        # Save a direct copy of the raw sensor data (without the parsed timestamp column)
        df_to_save = df.drop(columns=['timestamp_dt'])
        df_to_save.to_csv(sensor_master, index=False)
        logging.info(f"Updated sensor master dataset: {len(df)} total rows.")
        
        # Sort by timestamp for merge_asof
        df = df.dropna(subset=['timestamp_dt']).sort_values('timestamp_dt')
        return df
        
    except Exception as e:
        logging.error(f"Error reading sensor CSV: {e}")
        return None

# ---------------------------------------------------------
# IMAGE VALIDATION
# ---------------------------------------------------------

def validate_image(image_path):
    # Only validate existence and basic readability without full OpenCV if possible
    # We will just check file size for now to avoid heavy dependencies unless requested
    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        return 1
    return 0

# ---------------------------------------------------------
# CAMERA CAPTURE PROCESSING
# ---------------------------------------------------------

def discover_camera_captures(captures_dir):
    captures = []
    if not os.path.exists(captures_dir):
        return captures
        
    for item in os.listdir(captures_dir):
        capture_path = os.path.join(captures_dir, item)
        if os.path.isdir(capture_path):
            metadata_file = os.path.join(capture_path, "metadata.json")
            if os.path.exists(metadata_file):
                captures.append(metadata_file)
    
    # Sort chronologically by folder name (timestamp)
    captures.sort()
    return captures

def process_camera_capture(metadata_file, sensor_df, config):
    try:
        with open(metadata_file, 'r') as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Corrupted metadata file: {metadata_file}")
        return None
        
    # Generate deterministic capture ID
    section = meta.get('section_id', config['section_id'])
    channel = meta.get('channel_id', config['channel_id'])
    capture_time_str = meta.get('timestamp')
    
    if not capture_time_str:
        logging.warning(f"No timestamp found in {metadata_file}")
        return None
        
    capture_id = f"{section}_{channel}_{capture_time_str}"
    
    try:
        # Camera timestamp format is YYYYMMDD_HHMMSS
        capture_dt = datetime.strptime(capture_time_str, "%Y%m%d_%H%M%S")
    except ValueError:
        logging.error(f"Invalid timestamp format in {metadata_file}: {capture_time_str}")
        return None
        
    plants = meta.get('plants', [])
    if not plants:
        logging.info(f"No plants found in capture {capture_id}")
        return capture_id, []
        
    # Convert plants to DataFrame
    rows = []
    for p in plants:
        plant_id = p.get('plant_id')
        img_path = p.get('image_path')
        
        # Validation
        img_exists = validate_image(img_path)
        
        row = {
            'capture_id': capture_id,
            'timestamp': capture_time_str,
            'timestamp_dt': capture_dt,
            'experiment_id': config['experiment_id'],
            'crop': meta.get('crop', config['crop']),
            'section_id': section,
            'channel_id': channel,
            'plant_id': plant_id,
            'image_path': img_path,
            'image_exists': img_exists,
            # Initialize empty labels
            'health_label': '',
            'growth_stage_label': '',
            'nutrient_status_label': ''
        }
        rows.append(row)
        
    camera_df = pd.DataFrame(rows)
    camera_df = camera_df.sort_values('timestamp_dt')
    
    # -------------------------------------------------
    # SYNCHRONIZE WITH SENSORS
    # -------------------------------------------------
    
    if sensor_df is not None and not sensor_df.empty:
        # Perform nearest match within tolerance
        tolerance = pd.Timedelta(seconds=config['sensor_match_tolerance_seconds'])
        
        # merge_asof requires both dataframes to be sorted by the key
        merged_df = pd.merge_asof(
            camera_df,
            sensor_df,
            on='timestamp_dt',
            direction='nearest',
            tolerance=tolerance,
            suffixes=('', '_sensor')
        )
        
        # Add matching metadata
        matched_mask = merged_df['timestamp_sensor'].notna() if 'timestamp_sensor' in merged_df.columns else merged_df['ph'].notna()
        
        merged_df['sensor_match_status'] = 'NO_MATCH'
        merged_df.loc[matched_mask, 'sensor_match_status'] = 'MATCHED'
        
        merged_df['missing_sensor_flag'] = 1
        merged_df.loc[matched_mask, 'missing_sensor_flag'] = 0
        
        # Calculate time difference
        if 'timestamp_sensor' in merged_df.columns:
            merged_df = merged_df.rename(columns={'timestamp_sensor': 'sensor_timestamp'})
            sensor_dt = pd.to_datetime(merged_df['sensor_timestamp'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
            merged_df['time_difference_seconds'] = (merged_df['timestamp_dt'] - sensor_dt).dt.total_seconds().abs()
        else:
            merged_df['sensor_timestamp'] = ''
            merged_df['time_difference_seconds'] = ''
            
    else:
        # No sensor data at all
        merged_df = camera_df.copy()
        merged_df['sensor_match_status'] = 'NO_MATCH'
        merged_df['missing_sensor_flag'] = 1
        merged_df['sensor_timestamp'] = ''
        merged_df['time_difference_seconds'] = ''
        
    # Drop internal datetime column
    merged_df = merged_df.drop(columns=['timestamp_dt'], errors='ignore')
    
    return capture_id, merged_df

def append_master_rows(df, master_file):
    os.makedirs(os.path.dirname(master_file), exist_ok=True)
    
    # If file exists, append without header
    header = not os.path.exists(master_file)
    df.to_csv(master_file, mode='a', header=header, index=False)

# ---------------------------------------------------------
# MAIN WORKFLOW
# ---------------------------------------------------------

def process_once(config, state):
    logging.info("Starting Data Manager iteration...")
    
    # 1. Update Sensor Master Dataset
    sensor_df = update_sensor_master_dataset(config)
    
    # 2. Discover Camera Captures
    metadata_files = discover_camera_captures(config['camera_captures_dir'])
    logging.info(f"Discovered {len(metadata_files)} camera captures total.")
    
    processed_count = 0
    plant_images_count = 0
    match_count = 0
    unmatch_count = 0
    skipped_count = 0
    
    for meta_file in metadata_files:
        folder_name = os.path.basename(os.path.dirname(meta_file))
        capture_id = f"{config['section_id']}_{config['channel_id']}_{folder_name}"
        
        if capture_id in state['processed_captures']:
            skipped_count += 1
            continue
            
        logging.info(f"Processing new capture: {capture_id}")
        
        result = process_camera_capture(meta_file, sensor_df, config)
        if result is None:
            continue
            
        actual_capture_id, capture_df = result
        
        if len(capture_df) > 0:
            append_master_rows(capture_df, config['master_dataset'])
            
            # Update stats
            plant_images_count += len(capture_df)
            matches = (capture_df['sensor_match_status'] == 'MATCHED').sum()
            match_count += matches
            unmatch_count += (len(capture_df) - matches)
            
        # Update state
        state['processed_captures'].append(actual_capture_id)
        save_state(state, config['state_file'])
        processed_count += 1
        
    logging.info("============================================")
    logging.info("DATA MANAGER SUMMARY")
    logging.info("============================================")
    if sensor_df is not None:
        logging.info(f"Sensor observations processed: {len(sensor_df)}")
    logging.info(f"Camera captures discovered: {len(metadata_files)}")
    logging.info(f"Camera captures processed: {processed_count}")
    logging.info(f"Plant images processed: {plant_images_count}")
    logging.info(f"Sensor matches: {match_count}")
    logging.info(f"Sensor unmatched: {unmatch_count}")
    logging.info(f"Duplicate captures skipped: {skipped_count}")
    logging.info("============================================")
    
    return state

def watch_loop(config, state):
    interval = config.get('watch_interval_seconds', 5)
    logging.info(f"Starting Watch Mode. Interval: {interval} seconds.")
    
    try:
        while True:
            state = process_once(config, state)
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.info("Watch Mode interrupted by user. Shutting down.")

# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydroponic Data Manager")
    parser.add_argument("--once", action="store_true", help="Process currently available data once and exit")
    parser.add_argument("--watch", action="store_true", help="Run continuously and process new data as it arrives")
    args = parser.parse_args()
    
    config = load_config()
    setup_logging(config['log_file'])
    state = load_state(config['state_file'])
    
    logging.info("Application startup.")
    
    if args.watch:
        watch_loop(config, state)
    elif args.once:
        process_once(config, state)
    else:
        parser.print_help()
        
    logging.info("Application shutdown.")
