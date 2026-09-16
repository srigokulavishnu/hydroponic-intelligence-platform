from typing import Dict, Any, Optional

class ContextBuilder:
    def __init__(self, kb_loader, history_manager):
        """
        Initialize the Context Builder.
        
        Args:
            kb_loader: An instance of KBLoader (from backend.kb_loader).
            history_manager: An instance of HistoryManager (from backend.history_manager).
        """
        self.kb_loader = kb_loader
        self.history_manager = history_manager

    def summarize_current_state(self, sensor_state: Optional[Dict[str, Any]] = None, model_predictions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Return a structured summary of currently supplied sensor and prediction information.
        Does not invent missing fields.
        """
        return {
            "sensors": sensor_state if sensor_state is not None else {},
            "predictions": model_predictions if model_predictions is not None else {}
        }

    def summarize_history(self, plant_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Return a structured summary based ONLY on historical observations actually available.
        """
        latest = self.history_manager.get_latest_state(plant_id)
        recent = self.history_manager.get_recent_history(plant_id, limit=limit)
        
        return {
            "latest": latest if latest else {},
            "recent": recent if recent else []
        }

    def build_context(
        self,
        plant_id: str,
        crop_id: str = "PALAK",
        sensor_state: Optional[Dict[str, Any]] = None,
        model_predictions: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        stage_id: Optional[str] = None,
        history_limit: int = 10
    ) -> Dict[str, Any]:
        """
        Assembles all available evidence for a single plant into a structured context object.
        Retrieves matching KB entries and history, handling missing fields safely.
        """
        context_notes = []
        
        # 1. State Assembly
        current_state = self.summarize_current_state(sensor_state, model_predictions)
        
        # 2. Extract intelligence parameters for targeted KB retrieval
        active_stage_id = stage_id
        if not active_stage_id and model_predictions and model_predictions.get("growth"):
            active_stage_id = model_predictions["growth"].get("label")
            
        health_state = None
        if model_predictions and model_predictions.get("health"):
            health_state = model_predictions["health"].get("label")
            
        nutrient_status = None
        if model_predictions and model_predictions.get("nutrient"):
            nutrient_status = model_predictions["nutrient"].get("label")
            
        sensor_parameters = []
        if sensor_state:
            # Map basic parameters present in sensor data for knowledge retrieval context
            if "ph" in sensor_state: sensor_parameters.append("pH")
            if "ec" in sensor_state: sensor_parameters.append("EC")
            if "temperature" in sensor_state: sensor_parameters.append("Air temperature")
            if "humidity" in sensor_state: sensor_parameters.append("Relative humidity")
            if "ldr1" in sensor_state or "ldr2" in sensor_state: sensor_parameters.append("Light")

        # 3. Knowledge Base Retrieval
        stage_group = None
        if active_stage_id:
            stage_group = self.kb_loader.get_stage_group(active_stage_id)
            
        knowledge_res = self.kb_loader.get_relevant_knowledge(
            crop_id=crop_id,
            stage_id=active_stage_id,
            stage_group=stage_group,
            health_state=health_state,
            nutrient_status=nutrient_status,
            sensor_parameters=sensor_parameters,
            query=query,
            max_items=12
        )
        
        # 4. History Retrieval
        history_summary = self.summarize_history(plant_id, limit=history_limit)

        # 5. Deterministic Context Notes Generation
        if not model_predictions or not model_predictions.get("growth"):
            context_notes.append("Growth prediction is currently unavailable.")
            
        if not history_summary.get("latest"):
            context_notes.append("Plant has no historical observations.")
            
        match_status = knowledge_res.get("match_status", "NO_MATCH")
        context_notes.append(f"Knowledge search returned {match_status} matches.")
        
        # Append specific rules to notes so they show up in the UI "Why this answer?" dropdown
        for res in knowledge_res.get("results", []):
            content = res.get("content", {})
            if "question" in content and "answer" in content:
                context_notes.append(f"KB: {content['question']} -> {content['answer']}")
            elif "diagnosis" in content:
                context_notes.append(f"KB Diagnosis: {content['diagnosis']}")
            elif "action" in content:
                context_notes.append(f"KB Action: {content['action']}")
        
        # Assembly
        return {
            "plant": {
                "plant_id": plant_id,
                "crop_id": crop_id
            },
            "current_state": current_state,
            "history": history_summary,
            "knowledge": knowledge_res,
            "query": query if query else "",
            "context_notes": context_notes
        }

    def build_chat_context(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Wrapper around build_context to explicitly signify preparation 
        for a conversational LLM interaction layer.
        """
        return self.build_context(*args, **kwargs)


if __name__ == "__main__":
    import json
    from pathlib import Path
    
    # Import the actual dependencies for the test
    from kb_loader import KBLoader
    from history_manager import HistoryManager
    
    current_dir = Path(__file__).parent
    
    print("Initializing components for ContextBuilder test...")
    kb = KBLoader(kb_dir=str(current_dir.parent / "kb"))
    hm = HistoryManager(history_path=str(current_dir.parent / "history" / "plant_history.csv"))
    
    builder = ContextBuilder(kb_loader=kb, history_manager=hm)
    
    # Mock inputs
    target_plant = "C01_P17"
    
    sensor_data = {
        "timestamp": "2026-09-14T15:30:00Z",
        "ph": 6.2,
        "ec": 1.4,
        "temperature": 24.0,
        "humidity": 62.0,
        "ldr1": 700,
        "ldr2": 720
    }
    
    full_predictions = {
        "health": {
            "label": "STRESS",
            "confidence": 0.92
        },
        "growth": {
            "label": "G3",
            "confidence": 0.88
        },
        "nutrient": {
            "label": "NON_OPTIMAL",
            "confidence": 0.94
        }
    }
    
    missing_growth_predictions = {
        "health": {
            "label": "STRESS",
            "confidence": 0.92
        },
        "growth": None,
        "nutrient": {
            "label": "NON_OPTIMAL",
            "confidence": 0.94
        }
    }
    
    test_query = "Why does my plant look weak and yellow?"
    
    print("\n==================================================")
    print("TEST 1: Full Context Retrieval with Query")
    print("==================================================")
    
    context_1 = builder.build_context(
        plant_id=target_plant,
        sensor_state=sensor_data,
        model_predictions=full_predictions,
        query=test_query
    )
    
    print(f"Plant ID: {context_1['plant']['plant_id']}")
    print(f"Sensor values count: {len(context_1['current_state']['sensors'])}")
    print(f"Predictions supplied: {list(context_1['current_state']['predictions'].keys())}")
    print(f"History latest timestamp: {context_1['history']['latest'].get('timestamp', 'None')}")
    print(f"KB Match Status: {context_1['knowledge']['match_status']}")
    print(f"KB Results count: {len(context_1['knowledge']['results'])}")
    print("Context Notes:")
    for note in context_1["context_notes"]:
        print(f"  - {note}")
        
    print("\n==================================================")
    print("TEST 2: Context Retrieval with Missing Growth")
    print("==================================================")
    
    context_2 = builder.build_context(
        plant_id=target_plant,
        sensor_state=sensor_data,
        model_predictions=missing_growth_predictions,
        query=None
    )
    
    print("Context Notes (notice the unavailable growth note):")
    for note in context_2["context_notes"]:
        print(f"  - {note}")
    
    print(f"KB Match Status: {context_2['knowledge']['match_status']}")
    print(f"KB Results count: {len(context_2['knowledge']['results'])}")

    print("\n--- Test Complete ---")
