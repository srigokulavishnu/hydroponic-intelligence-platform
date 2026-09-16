import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

class HistoryManager:
    def __init__(self, history_path: str):
        """
        Initialize the History Manager.
        
        Args:
            history_path: Path to the plant_history.csv file.
        """
        self.history_path = Path(history_path)
        self.columns = [
            "timestamp", "plant_id", "health", "growth", "nutrient", 
            "ph", "ec", "action", "outcome", "notes"
        ]
        self._ensure_storage_exists()
        
    def _ensure_storage_exists(self):
        """Ensure the history directory and CSV file exist with the correct headers."""
        # Create directory if it doesn't exist
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create CSV with headers if it doesn't exist
        if not self.history_path.exists():
            with open(self.history_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def validate_plant_id(self, plant_id: str) -> bool:
        """
        Validate that the plant ID matches the allowed physical section range:
        C01_P01 through C01_P49
        """
        if not isinstance(plant_id, str):
            return False
        if not plant_id.startswith("C01_P"):
            return False
        
        try:
            num = int(plant_id.split("_P")[1])
            if 1 <= num <= 49:
                # Also ensure strict formatting if needed (e.g. padding C01_P01 vs C01_P1)
                # But allowing C01_P1 to C01_P49 as well for safety.
                return True
        except (IndexError, ValueError):
            pass
            
        return False

    def observation_exists(self, plant_id: str, timestamp: str) -> bool:
        """Check if an observation for a plant at a specific timestamp already exists."""
        if not self.history_path.exists():
            return False
            
        with open(self.history_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('plant_id') == plant_id and row.get('timestamp') == timestamp:
                    return True
        return False

    def save_observation(
        self, 
        timestamp: str, 
        plant_id: str, 
        health: Optional[str] = None, 
        growth: Optional[str] = None, 
        nutrient: Optional[str] = None, 
        ph: Optional[float] = None, 
        ec: Optional[float] = None, 
        action: Optional[str] = None, 
        outcome: Optional[str] = None, 
        notes: Optional[str] = None
    ) -> bool:
        """
        Save a new historical observation safely to the CSV.
        Validates plant ID and checks for duplicates.
        """
        if not self.validate_plant_id(plant_id):
            raise ValueError(f"Invalid plant ID: {plant_id}. Must be between C01_P01 and C01_P49.")
            
        if self.observation_exists(plant_id, timestamp):
            # Prevent duplicate identical rows
            return False
            
        row = {
            "timestamp": timestamp,
            "plant_id": plant_id,
            "health": health if health is not None else "",
            "growth": growth if growth is not None else "",
            "nutrient": nutrient if nutrient is not None else "",
            "ph": str(ph) if ph is not None else "",
            "ec": str(ec) if ec is not None else "",
            "action": action if action is not None else "",
            "outcome": outcome if outcome is not None else "",
            "notes": notes if notes is not None else ""
        }
        
        with open(self.history_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)
            
        return True

    def _parse_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Convert a raw CSV row dictionary into a typed state dictionary."""
        parsed = dict(row)
        
        # Safely convert numeric fields
        for field in ["ph", "ec"]:
            val = parsed.get(field, "").strip()
            if val:
                try:
                    parsed[field] = float(val)
                except ValueError:
                    parsed[field] = None
            else:
                parsed[field] = None
                
        return parsed

    def get_recent_history(self, plant_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent history for a plant, newest first."""
        if not self.validate_plant_id(plant_id) or not self.history_path.exists():
            return []
            
        observations = []
        with open(self.history_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('plant_id') == plant_id:
                    observations.append(self._parse_row(row))
                    
        # Sort newest first based on timestamp (lexicographical sort of ISO strings)
        observations.sort(key=lambda x: x["timestamp"], reverse=True)
        return observations[:limit]

    def get_latest_observation(self, plant_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the absolute newest observation for a plant."""
        recent = self.get_recent_history(plant_id, limit=1)
        if recent:
            return recent[0]
        return None

    def get_latest_state(self, plant_id: str) -> Optional[Dict[str, Any]]:
        """
        Alias for get_latest_observation providing a clean dictionary
        representing the latest known state fields.
        """
        return self.get_latest_observation(plant_id)

    def get_history_between(self, plant_id: str, start_timestamp: str, end_timestamp: str) -> List[Dict[str, Any]]:
        """Retrieve history bounded by two ISO timestamps."""
        if not self.validate_plant_id(plant_id) or not self.history_path.exists():
            return []
            
        observations = []
        with open(self.history_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('plant_id') == plant_id:
                    ts = row.get("timestamp", "")
                    # Direct string comparison works safely for consistent ISO formats
                    if start_timestamp <= ts <= end_timestamp:
                        observations.append(self._parse_row(row))
                        
        observations.sort(key=lambda x: x["timestamp"], reverse=True)
        return observations

    def get_all_plant_ids(self) -> List[str]:
        """Retrieve a list of all distinct plant IDs currently in the history."""
        if not self.history_path.exists():
            return []
            
        plant_ids = set()
        with open(self.history_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("plant_id")
                if pid:
                    plant_ids.add(pid)
                    
        return sorted(list(plant_ids))


if __name__ == "__main__":
    import os
    
    # Resolving absolute path to the history CSV
    current_dir = Path(__file__).parent
    history_file_path = current_dir.parent / "history" / "plant_history.csv"
    
    print(f"Initializing HistoryManager at: {history_file_path}")
    manager = HistoryManager(history_path=str(history_file_path))
    
    print("\n--- B/C. Testing Validation ---")
    valid_ids = ["C01_P01", "C01_P17", "C01_P49"]
    invalid_ids = ["C01_P00", "C01_P50", "P01", "PALAK_P01"]
    
    for pid in valid_ids:
        print(f"Validate {pid}: {'PASS' if manager.validate_plant_id(pid) else 'FAIL'}")
    for pid in invalid_ids:
        print(f"Validate {pid} (should fail): {'PASS (Correctly rejected)' if not manager.validate_plant_id(pid) else 'FAIL (Incorrectly accepted)'}")

    print("\n--- D. Saving Sample Observation ---")
    current_ts = datetime.utcnow().isoformat() + "Z"
    target_plant = "C01_P17"
    
    saved = manager.save_observation(
        timestamp=current_ts,
        plant_id=target_plant,
        health="STRESS",
        growth="G3",
        nutrient="NON_OPTIMAL",
        ph=6.2,
        ec=1.4,
        action="Monitor EC and inspect leaves",
        outcome="Pending",
        notes="Test observation"
    )
    print(f"Saved observation for {target_plant} at {current_ts}: {saved}")
    
    print("\n--- E. Testing Retrievals ---")
    latest = manager.get_latest_observation(target_plant)
    print(f"Latest observation timestamp: {latest['timestamp'] if latest else None}")
    
    recent = manager.get_recent_history(target_plant, limit=5)
    print(f"Recent history count for {target_plant}: {len(recent)}")
    
    state = manager.get_latest_state(target_plant)
    print(f"Latest state dump for {target_plant}:")
    for k, v in (state.items() if state else {}):
        print(f"  {k}: {v}")
        
    exists = manager.observation_exists(target_plant, current_ts)
    print(f"Observation exists validation: {exists}")
    
    # Try duplicate insertion
    duplicate_saved = manager.save_observation(
        timestamp=current_ts, 
        plant_id=target_plant,
        notes="Duplicate"
    )
    print(f"Duplicate save allowed? {duplicate_saved}")
    
    all_plants = manager.get_all_plant_ids()
    print(f"\nAll observed plant IDs: {all_plants}")
