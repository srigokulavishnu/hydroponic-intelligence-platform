import os
import json
import logging
from typing import Dict, Any, Optional, List, Union

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

class LLMService:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        Initializes the LLM service.
        Reads configuration safely without hardcoding secrets.
        """
        # Read from standard Gemini environment variable
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        # The model must be explicitly configured (e.g. from .env)
        self.model_name = model_name or os.environ.get("LLM_MODEL")
        
        self.system_prompt = system_prompt or (
            "You are the intelligence assistant for an NFT hydroponic monitoring system. "
            "You receive structured evidence from: current sensors, AI predictions, plant history, and the Knowledge Base.\n"
            "Use ONLY the supplied evidence. Never invent missing data.\n"
            "Clearly distinguish between measurements, predictions, historical observations, and reference knowledge.\n"
            "Growth predictions are currently unavailable if the model has not been trained.\n"
            "Nutrient predictions are stage-aware.\n"
            "When information is unavailable, say so clearly.\n"
            "If a user asks about sensor values, always evaluate them against the provided KB ranges.\n"
            "Do NOT issue actuator commands. Do NOT attempt to turn on pumps, fans, lighting, or any hardware.\n"
            "Do NOT claim certainty beyond the supplied evidence.\n"
            "Answer the user's question directly."
        )

        self.client = None
        if genai and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logging.error(f"Failed to initialize Gemini client: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Reports the operational status of the LLM configuration."""
        configured = bool(self.api_key and self.model_name)
        
        if not genai:
            status = "ERROR"
        elif not configured:
            status = "NOT_CONFIGURED"
        else:
            status = "READY"
            
        return {
            "provider": "google-genai",
            "model": self.model_name if self.model_name else "NOT_CONFIGURED",
            "configured": configured,
            "status": status,
            "sdk_installed": bool(genai)
        }

    def validate_context(self, context: Any) -> Dict[str, Any]:
        """Verifies the context object structure gracefully."""
        if not isinstance(context, dict):
            return {"valid": False, "warnings": ["Context must be a dictionary."]}
            
        warnings = []
        
        if "plant" not in context or not context.get("plant", {}).get("plant_id"):
            warnings.append("Missing plant ID in context.")
            
        if "current_state" not in context:
            warnings.append("Missing current_state section.")
        else:
            if "sensors" not in context["current_state"]:
                warnings.append("Missing sensors in current_state.")
            if "predictions" not in context["current_state"]:
                warnings.append("Missing predictions in current_state.")
                
        if "knowledge" not in context:
            warnings.append("Missing knowledge base section.")
            
        if "history" not in context:
            warnings.append("Missing history section.")

        return {
            "valid": True,  # We tolerate missing sections as long as it's a dict
            "warnings": warnings
        }

    def build_messages(self, context: Dict[str, Any], user_query: str) -> str:
        """Converts the structured context into a clear text prompt for the LLM."""
        
        context_str = "--- PLANT ---\n"
        plant_info = context.get("plant", {})
        context_str += json.dumps(plant_info, indent=2) + "\n\n"
        
        context_str += "--- CURRENT SENSORS ---\n"
        sensors = context.get("current_state", {}).get("sensors", {})
        if sensors:
            context_str += json.dumps(sensors, indent=2) + "\n\n"
        else:
            context_str += "No current sensor data available.\n\n"

        context_str += "--- CURRENT AI PREDICTIONS ---\n"
        predictions = context.get("current_state", {}).get("predictions", {})
        if predictions:
            context_str += json.dumps(predictions, indent=2) + "\n\n"
        else:
            context_str += "No AI predictions available.\n\n"

        context_str += "--- PLANT HISTORY ---\n"
        history = context.get("history", {})
        if history and (history.get("latest") or history.get("recent")):
            context_str += json.dumps(history, indent=2) + "\n\n"
        else:
            context_str += "The available history does not contain previous observations for this plant.\n\n"

        context_str += "--- KNOWLEDGE BASE ---\n"
        knowledge = context.get("knowledge", {})
        kb_status = knowledge.get("match_status", "NO_MATCH")
        if kb_status == "NO_MATCH":
            context_str += "Domain reference information is NO_MATCH (unavailable).\n\n"
        else:
            context_str += f"Status: {kb_status}\n"
            results = knowledge.get("results", [])
            context_str += json.dumps(results[:10], indent=2) + "\n\n" # Increased limit to 10 for better context
            
        context_str += "--- CONTEXT NOTES ---\n"
        notes = context.get("context_notes", [])
        if notes:
            for note in notes:
                context_str += f"- {note}\n"
        else:
            context_str += "No additional notes.\n"
            
        context_str += "\n--- USER QUESTION ---\n"
        context_str += user_query
        
        return context_str

    def generate_response(self, context: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """
        Sends the verified context to the LLM and returns the reasoned response.
        Handles missing configurations and API failures cleanly.
        """
        status_check = self.get_status()
        if status_check["status"] != "READY":
            return {
                "status": "ERROR",
                "response": None,
                "reason": f"LLM is not configured properly or SDK is missing. Status: {status_check['status']}"
            }
            
        val_result = self.validate_context(context)
        if not val_result["valid"]:
            return {
                "status": "ERROR",
                "response": None,
                "reason": f"Invalid context structure: {val_result.get('warnings', [])}"
            }

        # Build the structured context string
        context_str = self.build_messages(context, user_query)
        
        try:
            # We configure the Gemini request explicitly using types.GenerateContentConfig
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=context_str,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.3, # Low temperature for deterministic, factual reasoning
                    max_output_tokens=2048,
                )
            )
            
            content = response.text
            if not content:
                return {
                    "status": "ERROR",
                    "response": None,
                    "reason": "Received empty response from the LLM."
                }
                
            return {
                "status": "OK",
                "response": content,
                "model": self.model_name
            }
            
        except Exception as e:
            # Safely capture any generic API/network exception
            return {
                "status": "ERROR",
                "response": None,
                "reason": f"Gemini API error: {e}"
            }

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING LLM SERVICE (Gemini Integration Layer)")
    print("=" * 60)
    
    # Instantiate without live keys for dry-run testing
    service = LLMService(api_key="mock_key", model_name="mock_model")
    
    print("\nTEST 1 - SERVICE STATUS")
    print(json.dumps(service.get_status(), indent=2))
    
    mock_context = {
        "plant": {
            "plant_id": "C01_P17",
            "crop_id": "PALAK"
        },
        "current_state": {
            "sensors": {
                "pH": 6.2,
                "EC": 1.4,
                "temperature": 24.0,
                "humidity": 62.0,
                "light": 710
            },
            "predictions": {
                "health": {
                    "label": "STRESS",
                    "confidence": 0.92,
                    "status": "OK"
                },
                "growth": {
                    "status": "UNAVAILABLE",
                    "reason": "Growth model is not trained yet."
                },
                "nutrient": {
                    "status": "UNAVAILABLE",
                    "reason": "Growth stage is unavailable for the integrated stage-aware nutrient prediction."
                }
            }
        },
        "history": {
            "latest": {
                "timestamp": "2025-01-01 12:00",
                "health": "STRESS",
                "ph": 6.1
            },
            "recent": []
        },
        "knowledge": {
            "match_status": "DIRECT",
            "results": [
                {
                    "stage_id": "G3",
                    "stage_name": "Vegetative",
                    "keywords": ["Vegetative", "Biomass"]
                }
            ]
        },
        "context_notes": [
            "User wants to know why the plant looks weak."
        ]
    }
    
    print("\nTEST 2 - CONTEXT VALIDATION")
    print(json.dumps(service.validate_context(mock_context), indent=2))
    
    print("\nTEST 3 - MESSAGE BUILDING (Dry Run)")
    context_text = service.build_messages(mock_context, "Why does C01_P17 look weak?")
    print("  System Prompt:")
    print(f"    {service.system_prompt[:100]}...")
    print("  User Message Outline:")
    for section in context_text.split("--- "):
        if section.strip():
            print(f"    - {section.splitlines()[0]}")
            
    print("\nTEST 4 - MOCK RESPONSE")
    # Because we don't have a real API key in the environment, this will safely fail via the robust error handler
    # or via the SDK MISSING catch.
    if service.get_status()["status"] == "READY":
        res = service.generate_response(mock_context, "Why does C01_P17 look weak?")
        print(f"  Result Status: {res['status']}")
        if res['status'] == "ERROR":
            print(f"  Reason: {res['reason']}")
        else:
            print(f"  Response: {res['response']}")
    else:
        print("  Dry-run response test: OK (Gracefully bypassing API call due to missing SDK/Keys)")
        
    print("\nTEST 5 - LIVE API")
    live_service = LLMService()
    if live_service.get_status()["status"] == "READY":
        print("  Executing live inference...")
        live_res = live_service.generate_response(mock_context, "Why does C01_P17 look weak?")
        print(json.dumps(live_res, indent=2))
    else:
        print("  Live LLM test skipped because API configuration is not available.")
        
    print("\nTests complete.")
