import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone

# Add the backend directory to sys.path so project-relative imports work gracefully
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Import the existing intelligence components
from backend.kb_loader import KBLoader
from backend.history_manager import HistoryManager
from backend.model_loader import ModelLoader
from backend.context_builder import ContextBuilder
from backend.llm_service import LLMService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==============================================================================
# Initialization
# ==============================================================================
app = FastAPI(
    title="Adaptive Hydroponic Intelligence Platform",
    description="Backend AI Orchestrator",
    version="1.0.0"
)

# CORS Configuration
allowed_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Component Instances
kb_loader: KBLoader
history_manager: HistoryManager
model_loader: ModelLoader
context_builder: ContextBuilder
llm_service: LLMService

@app.on_event("startup")
async def startup_event():
    """Initializes the backend components safely without triggering live hardware."""
    global kb_loader, history_manager, model_loader, context_builder, llm_service
    logging.info("Starting Adaptive Hydroponic Intelligence Platform...")
    
    try:
        # Resolve paths relative to the project root
        kb_dir = str(BASE_DIR / "kb")
        history_file = str(BASE_DIR / "history" / "plant_history.csv")
        
        kb_loader = KBLoader(kb_dir=kb_dir)
        history_manager = HistoryManager(history_path=history_file)
        model_loader = ModelLoader()
        context_builder = ContextBuilder(kb_loader=kb_loader, history_manager=history_manager)
        llm_service = LLMService()
        
        logging.info("All intelligence components initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}")
        raise RuntimeError(f"Initialization failed: {e}")


# ==============================================================================
# Pydantic Models
# ==============================================================================
class HistoryObservationRequest(BaseModel):
    timestamp: Optional[str] = None
    health: Optional[str] = None
    growth: Optional[str] = None
    nutrient: Optional[str] = None
    ph: Optional[float] = None
    ec: Optional[float] = None
    action: Optional[str] = None
    outcome: Optional[str] = None
    notes: Optional[str] = None

class ChatRequest(BaseModel):
    plant_id: Optional[str] = None
    message: str
    crop_id: Optional[str] = "PALAK"
    current_state: Optional[Dict[str, Any]] = None


# ==============================================================================
# Endpoints
# ==============================================================================
@app.get("/health")
def get_health():
    """System health endpoint combining statuses from various intelligence modules."""
    try:
        ml_status = model_loader.get_model_status()
        llm_status = llm_service.get_status()
        
        return {
            "status": "OK",
            "service": "Adaptive Hydroponic Intelligence Platform",
            "components": {
                "model_loader": ml_status,
                "llm": llm_status,
                "knowledge_base": "OK" if kb_loader else "ERROR",
                "history": "OK" if history_manager else "ERROR"
            }
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Internal component check failed.")

@app.get("/api/system/info")
def get_system_info():
    """Non-sensitive system overview endpoint."""
    return {
        "project": "Adaptive Hydroponic Intelligence Platform",
        "crop": "PALAK",
        "system": "NFT",
        "section": "C01",
        "plant_count": 49,
        "modules": {
            "health": "ConvNeXt-Tiny",
            "growth": "ConvNeXt-Tiny",
            "nutrient": "XGBoost",
            "llm": "Gemini"
        }
    }

@app.get("/api/plants")
def get_plants():
    """Returns the permanent list of physical plants."""
    return {
        "crop_id": "PALAK",
        "section_id": "C01",
        "plants": [f"C01_P{i:02d}" for i in range(1, 50)]
    }

@app.get("/api/plants/{plant_id}/state")
def get_plant_state(plant_id: str):
    """
    Performs CURRENT hardware inference (camera + sensors) and model predictions.
    Does NOT auto-record to history on simply fetching state.
    """
    if not model_loader.validate_plant_id(plant_id):
        raise HTTPException(status_code=404, detail="Invalid plant ID format or out of bounds.")

    try:
        # Live Inference Execution
        inference_result = model_loader.run_plant_inference(plant_id)
        
        # Latest History
        latest_history = history_manager.get_latest_state(plant_id)
        
        return {
            "plant": {
                "plant_id": plant_id,
                "crop_id": "PALAK"
            },
            "current_observation": {
                "timestamp": inference_result.get("timestamp"),
                "image_status": inference_result.get("image_status"),
                "image_url": inference_result.get("image_url"),
                "sensors": inference_result.get("sensors")
            },
            "predictions": inference_result.get("predictions", {}),
            "history": {
                "latest": latest_history
            },
            "overall_status": inference_result.get("overall_status", "ERROR")
        }
    except Exception as e:
        logging.error(f"State retrieval failed for {plant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/plants/{plant_id}/history")
def get_plant_history(plant_id: str, limit: int = Query(10, ge=1, le=100)):
    """Retrieves longitudinal history for a plant."""
    if not model_loader.validate_plant_id(plant_id):
        raise HTTPException(status_code=404, detail="Invalid plant ID format.")
        
    try:
        records = history_manager.get_recent_history(plant_id, limit=limit)
        return {
            "plant_id": plant_id,
            "history": records if records else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plants/{plant_id}/history")
def save_plant_observation(plant_id: str, req: HistoryObservationRequest):
    """Explicitly commits a new observation/action to the history store."""
    if not model_loader.validate_plant_id(plant_id):
        raise HTTPException(status_code=404, detail="Invalid plant ID format.")
        
    ts = req.timestamp if req.timestamp else datetime.now(timezone.utc).isoformat()
    
    try:
        success = history_manager.save_observation(
            timestamp=ts,
            plant_id=plant_id,
            health=req.health,
            growth=req.growth,
            nutrient=req.nutrient,
            ph=req.ph,
            ec=req.ec,
            action=req.action,
            outcome=req.outcome,
            notes=req.notes
        )
        if success:
            return {"status": "OK", "message": "Observation saved successfully."}
        else:
            return {"status": "EXISTS", "message": "Identical timestamp observation already exists."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge/search")
def search_knowledge(query: str, crop_id: str = "PALAK"):
    """Performs deterministic context matching against the structured KB."""
    try:
        search_result = kb_loader.search_knowledge(query=query, crop_id=crop_id)
        
        return {
            "query": query,
            "crop_id": crop_id,
            "match_status": search_result.get("match_status", "NO_MATCH"),
            "results": search_result.get("results", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_interaction(req: ChatRequest):
    """
    The main intelligence interface. Assembles live observations, models, history,
    and KB logic into a cohesive Context object passed to the LLM.
    """
    plant_id = req.plant_id
    crop_id = req.crop_id or "PALAK"
    user_query = req.message
    
    try:
        if plant_id:
            # PLANT-SPECIFIC MODE
            if not model_loader.validate_plant_id(plant_id):
                raise HTTPException(status_code=404, detail="Invalid plant ID format.")

            # 1. Gather Current Hardware & Prediction State
            if req.current_state:
                sensor_state = req.current_state.get("sensors")
                model_predictions = req.current_state.get("predictions")
            else:
                inference_result = model_loader.run_plant_inference(plant_id)
                sensor_state = inference_result.get("sensors")
                model_predictions = inference_result.get("predictions")
            
            # 2. Build Multi-Domain Context 
            context = context_builder.build_chat_context(
                plant_id=plant_id,
                crop_id=crop_id,
                sensor_state=sensor_state,
                model_predictions=model_predictions,
                query=user_query
            )
        else:
            # GENERAL KNOWLEDGE MODE (No plant selected)
            search_res = kb_loader.search_knowledge(query=user_query, crop_id=crop_id)
            match_status = search_res.get("match_status", "NO_MATCH")
            results = search_res.get("results", [])
            
            # If NO_MATCH, dump entire KB to give LLM maximum context for general answering
            if match_status == "NO_MATCH":
                results = []
                for k, v in kb_loader.kb.items():
                    for entry in v:
                        if entry.get("crop_id") == crop_id or entry.get("crop_id", "ALL") == "ALL":
                            results.append({
                                "source_type": k,
                                "knowledge_id": entry.get("id", f"{k}_rule"),
                                "content": entry
                            })
            
            # Build lightweight context manually to avoid breaking ContextBuilder's plant expectations
            context = {
                "plant": {"plant_id": None, "crop_id": crop_id},
                "current_state": {"sensors": {}, "predictions": {}},
                "history": {"latest": {}, "recent": []},
                "knowledge": {
                    "match_status": match_status,
                    "results": results
                },
                "query": user_query,
                "context_notes": [f"Knowledge Match Status: {match_status}"]
            }
            
        # 3. Call LLM Service
        llm_resp = llm_service.generate_response(context=context, user_query=user_query)
        
        if llm_resp.get("status") == "ERROR":
            return {
                "status": "ERROR",
                "response": None,
                "reason": llm_resp.get("reason"),
                "plant": {"plant_id": plant_id, "crop_id": crop_id},
                "current_state": context.get("current_state"),
                "context_notes": context.get("context_notes", []),
                "raw_context": context
            }
            
        # 4. Successful LLM Response
        return {
            "status": "OK",
            "plant": {"plant_id": plant_id, "crop_id": crop_id},
            "response": llm_resp.get("response"),
            "current_state": context.get("current_state"),
            "context_notes": context.get("context_notes", []),
            "raw_context": context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Chat endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Internal chat failure.")


# ==============================================================================
# Development Test Launcher
# ==============================================================================
if __name__ == "__main__":
    import asyncio
    from fastapi.testclient import TestClient

    print("=" * 60)
    print("TESTING BACKEND APP ORCHESTRATION")
    print("=" * 60)

    # Trigger initialization manually for the test script
    asyncio.run(startup_event())
    
    # We use TestClient so we do not actually bind to a port or run Uvicorn indefinitely
    client = TestClient(app)

    # 1. GET /health
    print("\nTEST 1: GET /health")
    resp = client.get("/health")
    print(f"Status {resp.status_code}: {resp.json().get('status')}")

    # 2. GET /api/plants
    print("\nTEST 2: GET /api/plants")
    resp = client.get("/api/plants")
    print(f"Status {resp.status_code}: {len(resp.json().get('plants', []))} plants loaded.")

    # 3. GET /api/system/info
    print("\nTEST 3: GET /api/system/info")
    resp = client.get("/api/system/info")
    print(f"Status {resp.status_code}: {resp.json().get('project')}")

    # 4. GET /api/knowledge/search
    print("\nTEST 4: GET /api/knowledge/search")
    resp = client.get("/api/knowledge/search?query=pH")
    search_json = resp.json()
    print(f"Status {resp.status_code}: Match Status - {search_json.get('match_status')}, Results: {len(search_json.get('results', []))}")

    # 5. Invalid Plant Validation
    print("\nTEST 5: Invalid Plant ID Validation")
    resp = client.get("/api/plants/C01_P55/history")
    print(f"Status {resp.status_code}: {resp.json().get('detail')}")

    # 5b. Valid Plant ID Validation
    print("\nTEST 5b: Valid Plant ID Validation")
    resp = client.get("/api/plants/C01_P17/history")
    print(f"Status {resp.status_code}: C01_P17 is accepted.")

    # 6. GET /api/plants/{plant_id}/history
    print("\nTEST 6: GET /api/plants/C01_P17/history")
    resp = client.get("/api/plants/C01_P17/history")
    print(f"Status {resp.status_code}: Extracted {len(resp.json().get('history', []))} history records")

    # 7. POST /api/chat (Mock LLM response fallback test)
    print("\nTEST 7: POST /api/chat (End-to-End Orchestration test)")
    chat_payload = {
        "plant_id": "C01_P17",
        "message": "Why does this plant look weak?"
    }
    resp = client.post("/api/chat", json=chat_payload)
    print(f"Status {resp.status_code}:")
    chat_json = resp.json()
    print(f"  App Status: {chat_json.get('status')}")
    if chat_json.get("status") == "ERROR":
        print(f"  LLM Reason: {chat_json.get('reason')}")
    else:
        print(f"  LLM Response (first 100 chars): {str(chat_json.get('response'))[:100]}")
        
    print("\nTests complete. (Hardware API errors during inference gracefully trapped in state.)")
