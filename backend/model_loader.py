import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Setup path mapping for existing subsystems
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "data_collection" / "camera_logger"))
sys.path.append(str(BASE_DIR / "data_collection" / "sensor_logger"))

try:
    import stream_capture  # type: ignore
except ImportError:
    stream_capture = None

try:
    import sensor_logger  # type: ignore
except ImportError:
    sensor_logger = None

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from torchvision import transforms, models  # type: ignore
    from PIL import Image  # type: ignore
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:
    torch = None

try:
    import joblib  # type: ignore
    import xgboost as xgb  # type: ignore
except ImportError:
    joblib = None


class ModelLoader:
    def __init__(self):
        self.health_model_path = str(BASE_DIR / "models" / "plant-health-intelligence" / "saved_model" / "best_convnext_tiny.pth")
        self.nutrient_model_path = str(BASE_DIR / "models" / "adaptive-nutrient-intelligence" / "saved_models" / "nutrient_xgboost_stage_aware_baseline.pkl")
        
        self.health_model = None
        self.health_classes = []
        
        self.nutrient_model = None
        
        # Exact training order verified from model_metadata.json and Jupyter Notebook
        self.nutrient_features = [
            "ph",
            "ec",
            "temperature",
            "humidity",
            "light",
            "growth_stage_encoded"
        ]
        
        # Verify from XGBoost model_metadata.json
        self.nutrient_classes = ["NON_OPTIMAL", "OPTIMAL"]
        
        # Verified from Jupyter Notebook
        self.growth_stage_mapping = {
            "EARLY": 0,
            "MIDDLE": 1,
            "LATE": 2
        }
        
        self._load_models()

    def _load_models(self):
        """Safely load checkpoints from disk."""
        # 1. Health Model
        if torch and Path(self.health_model_path).exists():
            try:
                checkpoint = torch.load(self.health_model_path, map_location='cpu', weights_only=False)
                
                # Check for explicit class names embedded in the checkpoint
                self.health_classes = checkpoint.get('class_names', [])
                if not self.health_classes:
                    logging.error("Health checkpoint lacks 'class_names'.")
                    raise ValueError("Missing class mapping.")
                    
                try:
                    import timm
                    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=len(self.health_classes))
                except ImportError:
                    # Fallback to torchvision if timm isn't installed
                    model = models.convnext_tiny(weights=None)
                    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(self.health_classes))
                
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint)))
                model.eval()
                self.health_model = model
                
                # Exact preprocessing for ConvNeXt
                image_size = checkpoint.get('image_size', 224)
                self.health_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
            except Exception as e:
                logging.error(f"Failed to load Health model: {e}")
                self.health_model = "ERROR"
        else:
            self.health_model = "UNAVAILABLE"
            
        # 2. Nutrient Model
        if joblib and Path(self.nutrient_model_path).exists():
            try:
                self.nutrient_model = joblib.load(self.nutrient_model_path)
            except Exception as e:
                logging.error(f"Failed to load Nutrient model: {e}")
                self.nutrient_model = "ERROR"
        else:
            self.nutrient_model = "UNAVAILABLE"

    def get_model_status(self) -> Dict[str, Any]:
        """Report on the operational status of the inference endpoints."""
        return {
            "health": {
                "loaded": isinstance(self.health_model, torch.nn.Module) if torch else False,
                "model_type": "ConvNeXt-Tiny",
                "path": self.health_model_path if Path(self.health_model_path).exists() else None,
                "status": "OK" if isinstance(self.health_model, torch.nn.Module) else self.health_model
            },
            "growth": {
                "loaded": False,
                "model_type": "ConvNeXt-Tiny",
                "path": None,
                "status": "UNAVAILABLE",
                "reason": "Growth model is not trained yet."
            },
            "nutrient": {
                "loaded": self.nutrient_model is not None and self.nutrient_model != "ERROR" and self.nutrient_model != "UNAVAILABLE",
                "model_type": "XGBoost",
                "path": self.nutrient_model_path if Path(self.nutrient_model_path).exists() else None,
                "status": "OK" if (self.nutrient_model is not None and self.nutrient_model not in ("ERROR", "UNAVAILABLE")) else self.nutrient_model
            }
        }

    def validate_plant_id(self, plant_id: str) -> bool:
        """Enforces strictly formatted plant IDs (C01_P01 through C01_P49)."""
        if not isinstance(plant_id, str):
            return False
        if not plant_id.startswith("C01_P"):
            return False
        try:
            num = int(plant_id.split("_P")[1])
            if 1 <= num <= 49 and len(plant_id) == 7:
                return True
        except ValueError:
            return False
        return False

    def get_current_plant_image(self, plant_id: str):
        """
        Uses the existing Stream Capture logic to fetch the image crop.
        It perfectly emulates the cv2.perspectiveTransform cropping math used in 
        stream_capture.save_capture to yield a native square crop without saving it.
        """
        if not self.validate_plant_id(plant_id):
            return None, "Invalid plant ID"
            
        if stream_capture is None or not torch:
            return None, "Camera modules unavailable."

        # Re-use the existing camera configuration exclusively
        try:
            stream_capture.load_config()
            holes = stream_capture.calibrated_holes
            roi = stream_capture.roi_points
            padding = stream_capture.CROP_PADDING
            
            # Extract CAMERA_URL directly from the stream_capture.py source code
            # since it is defined under if __name__ == "__main__": and not exported.
            camera_url = None
            with open(stream_capture.__file__, 'r') as f:
                for line in f:
                    if 'CAMERA_URL' in line and '=' in line and 'rtsp://' in line:
                        camera_url = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
            if not camera_url:
                raise AttributeError("CAMERA_URL not found in stream_capture.py source.")
                
        except Exception as e:
            return None, f"Failed to load stream_capture configuration: {e}"

        target_hole = next((h for h in holes if h['id'] == plant_id), None)
        if not target_hole:
            return None, "Plant ID not found in calibrated holes mapping."

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"
        cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None, "Could not connect to RTSP stream."

        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return None, "Failed to read frame from stream."

        # The exact inverse perspective math adapted directly from stream_capture.save_capture()
        warped, M = stream_capture.warp_channel(frame, roi)
        M_inv = np.linalg.inv(M)

        x, y, w, h_box = target_hole['x'], target_hole['y'], target_hole['w'], target_hole['h']
        cw_x = x + w / 2.0
        cw_y = y + h_box / 2.0
        
        # map warped center -> original camera frame
        center_warped = np.array([[[cw_x, cw_y]]], dtype=np.float32)
        center_native = cv2.perspectiveTransform(center_warped, M_inv)

        native_x = int(round(center_native[0, 0, 0]))
        native_y = int(round(center_native[0, 0, 1]))

        # Calculate the true native radius
        pts_warped = np.array([
            [[cw_x, cw_y]],            # center
            [[x, cw_y]],               # left edge
            [[x + w, cw_y]],           # right edge
            [[cw_x, y]],               # top edge
            [[cw_x, y + h_box]]        # bottom edge
        ], dtype=np.float32)

        pts_native = cv2.perspectiveTransform(pts_warped, M_inv)
        nc = pts_native[0, 0]
        max_radius = max(
            np.linalg.norm(pts_native[1, 0] - nc),
            np.linalg.norm(pts_native[2, 0] - nc),
            np.linalg.norm(pts_native[3, 0] - nc),
            np.linalg.norm(pts_native[4, 0] - nc)
        )

        side = int(np.ceil(2 * max_radius)) + 2 * padding
        half_side = side // 2

        x1 = max(0, native_x - half_side)
        y1 = max(0, native_y - half_side)
        x2 = min(frame.shape[1], native_x + half_side)
        y2 = min(frame.shape[0], native_y + half_side)

        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            
            # Pad if we hit image borders
            h_crop, w_crop = crop.shape[:2]
            if h_crop != side or w_crop != side:
                square_crop = np.zeros((side, side, 3), dtype=np.uint8)
                x_off = (side - w_crop) // 2
                y_off = (side - h_crop) // 2
                square_crop[y_off:y_off+h_crop, x_off:x_off+w_crop] = crop
                crop = square_crop
                
            crop = stream_capture.enhance_image(crop)
            # Convert OpenCV (BGR) to PIL (RGB)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            return pil_img, "OK"

        return None, "Invalid crop dimensions"

    def get_current_sensor_values(self) -> Optional[Dict[str, Any]]:
        """Uses the existing Sensor Logger to fetch current readings."""
        if sensor_logger is None:
            return None
            
        raw_response = sensor_logger.fetch_sensor_data()
        if raw_response is None:
            return None
            
        validated_data = sensor_logger.validate_api_response(raw_response)
        if validated_data is None:
            return None
            
        record = sensor_logger.extract_sensor_record(validated_data)
        
        # Standardize for the Nutrient Model schema
        # Missing numeric values gracefully decay to None rather than fake data
        def parse_float(val):
            try:
                return float(val) if val != "" else None
            except (ValueError, TypeError):
                return None
                
        # The sensor API returns EC in µS/cm (e.g. 1086), but models and KB expect mS/cm (e.g. 1.086)
        ec_val = parse_float(record.get("ec"))
        if ec_val is not None:
            ec_val = round(ec_val / 1000.0, 3)
                
        return {
            "temperature": parse_float(record.get("temperature_avg", record.get("temperature1"))),
            "humidity": parse_float(record.get("humidity_avg", record.get("humidity1"))),
            "light": parse_float(record.get("ldr1")),
            "ph": parse_float(record.get("ph")),
            "ec": ec_val
        }

    def predict_health(self, image) -> Dict[str, Any]:
        """Predicts plant health condition using ConvNeXt-Tiny."""
        if isinstance(self.health_model, str):
            return {
                "label": None,
                "confidence": None,
                "probabilities": {},
                "status": self.health_model,
                "reason": "Health model not fully loaded."
            }
            
        if image is None:
            return {
                "label": None,
                "confidence": None,
                "probabilities": {},
                "status": "ERROR",
                "reason": "No image provided."
            }

        try:
            input_tensor = self.health_transform(image).unsqueeze(0)
            with torch.no_grad():
                output = self.health_model(input_tensor)
                probs = F.softmax(output, dim=1)[0].tolist()
            
            prob_dict = {self.health_classes[i]: p for i, p in enumerate(probs)}
            max_idx = np.argmax(probs)
            confidence = probs[max_idx]
            label = self.health_classes[max_idx]
            
            return {
                "label": label,
                "confidence": float(confidence),
                "probabilities": prob_dict,
                "status": "OK"
            }
        except Exception as e:
            return {
                "label": None,
                "confidence": None,
                "probabilities": {},
                "status": "ERROR",
                "reason": f"Prediction failed: {e}"
            }

    def predict_growth(self, image) -> Dict[str, Any]:
        """Growth model is unavailable."""
        return {
            "label": None,
            "confidence": None,
            "probabilities": {},
            "status": "UNAVAILABLE",
            "reason": "Growth model is not trained yet."
        }

    def predict_nutrient(self, ph: float, ec: float, temperature: float, humidity: float, light: float, growth_stage: str) -> Dict[str, Any]:
        """
        Predicts optimal/non-optimal nutrient conditions based on sensor inputs and growth stage.
        Strict feature vector mapped matching exact training order: [ph, ec, temperature, humidity, light, growth_stage_encoded]
        """
        if self.nutrient_model is None or isinstance(self.nutrient_model, str):
            return {
                "label": None,
                "confidence": None,
                "probabilities": {},
                "status": self.nutrient_model if isinstance(self.nutrient_model, str) else "UNAVAILABLE",
                "reason": "Nutrient model not successfully loaded."
            }
            
        if not growth_stage:
            return {
                "label": None,
                "confidence": None,
                "probabilities": {},
                "status": "UNAVAILABLE",
                "reason": "Growth stage is unavailable for the integrated stage-aware nutrient prediction."
            }
            
        # Exact encoding mapping strictly adhered to
        encoded_stage = None
        if growth_stage in ["G0", "G1"]:
            encoded_stage = self.growth_stage_mapping["EARLY"]
        elif growth_stage == "G2":
            encoded_stage = self.growth_stage_mapping["MIDDLE"]
        elif growth_stage in ["G3", "G4"]:
            encoded_stage = self.growth_stage_mapping["LATE"]
            
        if encoded_stage is None:
            return {
                "label": None,
                "confidence": None,
                "probabilities": {},
                "status": "ERROR",
                "reason": f"Invalid growth stage provided: {growth_stage}"
            }

        # Null feature enforcement
        features_dict = {
            "ph": ph, "ec": ec, "temperature": temperature, 
            "humidity": humidity, "light": light, 
            "growth_stage_encoded": encoded_stage
        }
        
        for k, v in features_dict.items():
            if v is None:
                return {
                    "label": None,
                    "confidence": None,
                    "probabilities": {},
                    "status": "ERROR",
                    "reason": f"Missing required sensor feature: {k}"
                }

        try:
            # Build strictly ordered feature vector based on notebook verification
            feature_vector = [features_dict[feat] for feat in self.nutrient_features]
            X = np.array([feature_vector])
            
            # Predict
            pred_idx = int(self.nutrient_model.predict(X)[0])
            label = self.nutrient_classes[pred_idx]
            
            confidence = None
            prob_dict = {}
            if hasattr(self.nutrient_model, "predict_proba"):
                probs = self.nutrient_model.predict_proba(X)[0].tolist()
                prob_dict = {self.nutrient_classes[i]: p for i, p in enumerate(probs)}
                confidence = probs[pred_idx]
                
            return {
                "label": label,
                "confidence": confidence,
                "probabilities": prob_dict,
                "status": "OK"
            }
            
        except Exception as e:
            return {
                "label": None,
                "confidence": None,
                "probabilities": {},
                "status": "ERROR",
                "reason": f"XGBoost inference failed: {e}"
            }

    def run_plant_inference(self, plant_id: str) -> Dict[str, Any]:
        """Orchestrates the entire intelligence loop using project subsystems."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        if not self.validate_plant_id(plant_id):
            return {
                "plant_id": plant_id,
                "timestamp": timestamp,
                "overall_status": "ERROR",
                "reason": "Invalid plant ID."
            }

        # 1. Image Capture Observation
        image, img_status = self.get_current_plant_image(plant_id)
        
        health_pred = {"status": "ERROR", "reason": img_status}
        growth_pred = {"status": "UNAVAILABLE", "reason": "Growth model is not trained yet."}
        
        if image:
            health_pred = self.predict_health(image)
            growth_pred = self.predict_growth(image)
            
        # 2. Sensor API Observation
        sensor_data = self.get_current_sensor_values()
        
        if not sensor_data:
            nutrient_pred = {"status": "ERROR", "reason": "Failed to fetch or validate sensor API data."}
        else:
            # 3. Nutrient Prediction (Fails elegantly due to missing growth stage)
            nutrient_pred = self.predict_nutrient(
                ph=sensor_data.get("ph"),
                ec=sensor_data.get("ec"),
                temperature=sensor_data.get("temperature"),
                humidity=sensor_data.get("humidity"),
                light=sensor_data.get("light"),
                growth_stage=growth_pred.get("label")
            )

        overall = "OK"
        if health_pred["status"] != "OK" or nutrient_pred["status"] not in ("OK", "UNAVAILABLE"):
            overall = "ERROR"
        elif nutrient_pred["status"] == "UNAVAILABLE" or growth_pred["status"] == "UNAVAILABLE":
            overall = "PARTIAL"

        image_url = None
        if image:
            import base64
            from io import BytesIO
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{img_str}"

        return {
            "plant_id": plant_id,
            "timestamp": timestamp,
            "overall_status": overall,
            "image_status": img_status,
            "image_url": image_url,
            "sensors": sensor_data,
            "predictions": {
                "health": health_pred,
                "growth": growth_pred,
                "nutrient": nutrient_pred
            }
        }


if __name__ == "__main__":
    import json
    
    loader = ModelLoader()
    
    print("=" * 60)
    print("TESTING MODEL LOADER (Integrated Intelligence Pipeline)")
    print("=" * 60)
    
    print("\nTEST 1 - MODEL STATUS")
    print(json.dumps(loader.get_model_status(), indent=2))
    
    print("\nTEST 2 - PLANT ID VALIDATION")
    test_ids = ["C01_P01", "C01_P17", "C01_P49", "C01_P00", "C01_P50", "P01", "PALAK_P01"]
    for pid in test_ids:
        print(f"  {pid}: {'VALID' if loader.validate_plant_id(pid) else 'INVALID'}")
        
    print("\nTEST 3 - NUTRIENT DIRECT TEST (G3 explicitly supplied)")
    print(f"  Verified Feature Order: {loader.nutrient_features}")
    print(f"  Supplied Growth Stage: G3 -> Mapping to Group: LATE -> Encoded as: {loader.growth_stage_mapping['LATE']}")
    res3 = loader.predict_nutrient(ph=6.2, ec=1.4, temperature=24.0, humidity=62.0, light=710, growth_stage="G3")
    print("  Result:")
    print(json.dumps(res3, indent=2))
    
    print("\nTEST 4 - MISSING GROWTH STAGE IN NUTRIENT")
    res4 = loader.predict_nutrient(ph=6.2, ec=1.4, temperature=24.0, humidity=62.0, light=710, growth_stage=None)
    print(f"  Result Status: {res4['status']} (Reason: {res4.get('reason')})")
    
    print("\nTEST 5 - GROWTH MODEL (EXPECTED UNAVAILABLE)")
    res5 = loader.predict_growth(None)
    print(f"  Result Status: {res5['status']} (Reason: {res5.get('reason')})")
    
    print("\nTEST 6 - INTEGRATED PLANT INFERENCE")
    print("  Running inference for C01_P17. (May fail gracefully if hardware/RTSP disconnected).")
    res6 = loader.run_plant_inference("C01_P17")
    print(f"  Timestamp: {res6.get('timestamp')}")
    print(f"  Overall Status: {res6.get('overall_status')}")
    print(f"  Image Status: {res6.get('image_status')}")
    print(f"  Sensors: {res6.get('sensors')}")
    print("  Predictions:")
    print(f"    Health: {res6['predictions']['health']['status']} - {res6['predictions']['health'].get('reason', '')}")
    print(f"    Growth: {res6['predictions']['growth']['status']} - {res6['predictions']['growth'].get('reason', '')}")
    print(f"    Nutrient: {res6['predictions']['nutrient']['status']} - {res6['predictions']['nutrient'].get('reason', '')}")
    print("\nTests complete.")
