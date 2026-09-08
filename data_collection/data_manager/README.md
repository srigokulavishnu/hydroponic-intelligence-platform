# Hydroponic Data Manager

The Data Manager is a server-side daemon responsible for continuously organizing sensor observations and camera captures into a synchronized Master Dataset. It prepares clean, highly-organized source data for the future Intelligence Models (Growth, Health, Nutrient).

## Architecture

1. **Sensor Logger**: Collects environmental values approximately every 1 minute.
2. **Camera Logger**: Collects 1 clean image per plant approximately every 30 minutes.
3. **Data Manager**: Runs continuously to discover new camera captures, synchronize them with the closest sensor reading, and output exactly one row per plant into `master_dataset.csv`.

## Data Sources & Safety

The Data Manager treats the source data (`sensor_environment.csv` and camera captures) as **strictly raw and read-only**. It will never mutate, delete, or overwrite the original source files.

It generates two output datasets:
1. `sensor_master_dataset.csv`: A pristine, consolidated copy of all sensor readings.
2. `master_dataset.csv`: The synchronized image-to-sensor dataset.

## Installation & Deployment

This tool is designed to run 24/7 via SSH on a generic Linux server.

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hydroponic-intelligence-platform/hydroponic_data_manager
   ```

2. **Setup Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Settings**
   Ensure `config.json` points to the correct absolute or relative paths for your sensor log and datasets. 

4. **Testing (Once Mode)**
   Run the processor once to verify configuration:
   ```bash
   python data_manager.py --once
   ```

5. **Daemon Mode (Watch Mode)**
   Run the processor continuously:
   ```bash
   python data_manager.py --watch
   ```

   **To run persistently on a server:**
   ```bash
   nohup python data_manager.py --watch > logs/nohup.log 2>&1 &
   ```
   Or set it up as a `systemd` service for robust restart capabilities.

## Data Schema Rules

- **Permanent Plant IDs**: Maintained directly from the camera metadata.
- **Sensor Nulls**: Values like `NO-COMM` are propagated directly to the dataset instead of being zeroed.
- **Labels**: Initial values for `health_label`, `growth_stage_label`, and `nutrient_status_label` are left blank, awaiting the downstream labeling pipeline.
