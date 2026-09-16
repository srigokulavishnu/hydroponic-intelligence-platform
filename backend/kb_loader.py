import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

class KBLoader:
    def __init__(self, kb_dir: str):
        """
        Initialize the Knowledge Base loader.
        
        Args:
            kb_dir: Path to the directory containing the KB JSON files.
        """
        self.kb_dir = Path(kb_dir)
        self.kb = {
            "crop": [],
            "growth": [],
            "nutrient": [],
            "health": []
        }
        
        # Project baseline mapping from exact growth stage to nutrient stage group
        self.stage_mapping = {
            "G0": "EARLY",
            "G1": "EARLY",
            "G2": "MIDDLE",
            "G3": "LATE",
            "G4": "LATE"
        }
        
        # Domain-aware synonym expansion map
        self.synonym_map = {
            "weak": ["stress", "wilting", "reduced canopy", "abnormal development"],
            "yellow": ["yellowing", "chlorosis", "discoloration"],
            "yellowing": ["chlorosis", "discoloration", "stress"],
            "drooping": ["wilting", "stress"],
            "wilting": ["drooping", "stress"],
            "damaged": ["damage", "necrosis", "leaf damage"],
            "dead": ["necrosis", "leaf damage"],
            "hot": ["temperature", "heat stress"],
            "too hot": ["temperature", "heat stress"],
            "cold": ["temperature"],
            "nutrient problem": ["nutrient", "EC", "pH", "NON_OPTIMAL"],
            "nutrition": ["nutrient", "EC"],
            "acidic": ["pH"],
            "alkaline": ["pH"],
            "light": ["LDR", "PPFD", "DLI"],
            "plant looks bad": ["stress", "damage", "wilting", "yellowing"]
        }
        
        # Sensor parameter normalization map
        self.parameter_map = {
            "ph": "pH",
            "ec": "EC",
            "electrical conductivity": "EC",
            "temperature": "Air temperature", 
            "air temperature": "Air temperature",
            "humidity": "Relative humidity",
            "relative humidity": "Relative humidity",
            "light": "Light",
            "ldr": "Light"
        }
        
        # Nutrient status normalization map
        self.nutrient_status_map = {
            "optimal": "OPTIMAL",
            "non-optimal": "NON_OPTIMAL",
            "non optimal": "NON_OPTIMAL",
            "non_optimal": "NON_OPTIMAL"
        }
        
        self.load_knowledge_base()

    def _load_json(self, filename: str) -> List[Dict[str, Any]]:
        """Utility to load a JSON file and validate it contains an array."""
        file_path = self.kb_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Knowledge base file missing: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON in {filename}: {e}")
        
        if not isinstance(data, list):
            raise TypeError(f"Knowledge base {filename} must contain a JSON array at the top level.")
            
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise TypeError(f"Entry at index {idx} in {filename} is not a dictionary.")
            kid = entry.get("knowledge_id") or entry.get("id")
            if not kid:
                pass # Non-critical failure, handled gracefully in retrieval
            
        return data

    def load_knowledge_base(self):
        """Load all four KB files into memory."""
        self.kb["crop"] = self._load_json("crop_knowledge.json")
        self.kb["growth"] = self._load_json("growth_knowledge.json")
        self.kb["nutrient"] = self._load_json("nutrient_rules.json")
        self.kb["health"] = self._load_json("health_rules.json")

    def _normalize_text(self, text: str) -> List[str]:
        """Convert to lowercase, ignore punctuation, and collapse whitespace."""
        if not text:
            return []
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        stopwords = {'of', 'the', 'and', 'is', 'a', 'to', 'in', 'it', 'for', 'on', 'with', 'as', 'this', 'that', 'by', 'an', 'be', 'are', 'or', 'why', 'does', 'my'}
        return [word for word in clean_text.split() if word and word not in stopwords]

    def _normalize_value(self, val: str) -> str:
        """Standardize a field value for deterministic comparison."""
        if not val:
            return ""
        return val.strip().lower()

    def get_stage_group(self, stage_id: str) -> Optional[str]:
        """Map an exact stage_id (e.g. G3) to a nutrient stage group (e.g. LATE)."""
        if not stage_id:
            return None
        return self.stage_mapping.get(stage_id.upper())

    def get_crop_knowledge(self, crop_id: str = "PALAK", topic: Optional[str] = None, max_items: int = 10) -> List[Dict[str, Any]]:
        results = []
        for entry in self.kb["crop"]:
            e_crop = entry.get("crop_id", "") or entry.get("crop", "")
            if self._normalize_value(e_crop) != self._normalize_value(crop_id):
                continue
            if topic and self._normalize_value(entry.get("topic", "")) != self._normalize_value(topic):
                continue
            results.append(entry)
            if len(results) >= max_items:
                break
        return results

    def get_growth_knowledge(self, crop_id: str = "PALAK", stage_id: Optional[str] = None, max_items: int = 10) -> List[Dict[str, Any]]:
        results = []
        # Prioritize exact stage_id matches
        for entry in self.kb["growth"]:
            if self._normalize_value(entry.get("crop_id", "")) != self._normalize_value(crop_id):
                continue
            if stage_id and self._normalize_value(entry.get("stage_id", "")) == self._normalize_value(stage_id):
                results.append(entry)
                
        # Fill remaining with general growth entries if max_items not met
        for entry in self.kb["growth"]:
            if len(results) >= max_items:
                break
            if self._normalize_value(entry.get("crop_id", "")) != self._normalize_value(crop_id):
                continue
            if not stage_id or self._normalize_value(entry.get("stage_id", "")) != self._normalize_value(stage_id):
                if entry not in results:
                    results.append(entry)
                    
        return results[:max_items]

    def get_nutrient_knowledge(self, crop_id: str = "PALAK", stage_group: Optional[str] = None, parameter: Optional[str] = None, max_items: int = 10) -> List[Dict[str, Any]]:
        results = []
        
        # Normalize inputs
        norm_group = self._normalize_value(stage_group) if stage_group else None
        norm_param = self._normalize_value(parameter) if parameter else None
        if norm_param and norm_param in self.parameter_map:
            norm_param = self._normalize_value(self.parameter_map[norm_param])
            
        for entry in self.kb["nutrient"]:
            if self._normalize_value(entry.get("crop_id", "")) != self._normalize_value(crop_id):
                continue
            
            entry_stage = self._normalize_value(entry.get("stage_group", ""))
            # If a stage group is specified, and the entry has one, they must match.
            # If the entry is general (no stage group), it is eligible.
            if norm_group and entry_stage and entry_stage != norm_group:
                continue
                
            entry_param = self._normalize_value(entry.get("parameter", ""))
            if norm_param and entry_param and entry_param != norm_param:
                continue
                
            results.append(entry)
            if len(results) >= max_items:
                break
        return results

    def get_health_knowledge(self, crop_id: str = "PALAK", health_state: Optional[str] = None, indicator: Optional[str] = None, max_items: int = 10) -> List[Dict[str, Any]]:
        results = []
        norm_health = self._normalize_value(health_state) if health_state else None
        norm_ind = self._normalize_value(indicator) if indicator else None
            
        for entry in self.kb["health"]:
            if self._normalize_value(entry.get("crop_id", "")) != self._normalize_value(crop_id):
                continue
                
            if norm_health and self._normalize_value(entry.get("health_state", "")) != norm_health:
                continue
                
            if norm_ind and norm_ind not in self._normalize_value(entry.get("indicator", "")):
                continue
                
            results.append(entry)
            if len(results) >= max_items:
                break
        return results

    def _expand_query(self, query: str) -> tuple[set, set]:
        """Expands a query using the synonym_map. Returns (direct_words, related_words)."""
        direct_words = set(self._normalize_text(query))
        
        # Remove generic crop names to prevent spurious matches
        generic_crop_names = {'palak', 'spinach', 'spinacia', 'oleracea'}
        direct_words = direct_words - generic_crop_names
        
        query_lower = query.lower()
        related_words = set()
        
        for key_phrase, expansions in self.synonym_map.items():
            if key_phrase in query_lower:
                for exp in expansions:
                    related_words.update(self._normalize_text(exp))
                    
        # Exclude direct words and generic names from related words
        related_words = related_words - direct_words - generic_crop_names
        return direct_words, related_words

    def search_knowledge(self, query: str, crop_id: str = "PALAK", max_items: int = 10) -> Dict[str, Any]:
        """
        Search knowledge across all domains using deterministic keyword and related-term matching.
        Returns: {"match_status": "DIRECT"|"RELATED"|"NO_MATCH", "results": [...]}
        """
        direct_words, related_words = self._expand_query(query)
        if not direct_words and not related_words:
            return {"match_status": "NO_MATCH", "results": []}

        scored_results = {}
        has_direct_match = False
        
        # Priority definition for structured relevance
        priority_map = {"health": 4, "nutrient": 3, "growth": 2, "crop": 1}
        
        for kb_type, entries in self.kb.items():
            for entry in entries:
                e_crop = entry.get("crop_id", "") or entry.get("crop", "")
                if self._normalize_value(e_crop) != self._normalize_value(crop_id):
                    continue
                
                score = 0
                is_direct = False
                is_related = False
                
                # Extract structured fields
                h_state = self._normalize_text(entry.get("health_state", ""))
                s_id = self._normalize_text(entry.get("stage_id", ""))
                param = self._normalize_text(entry.get("parameter", ""))
                ind = self._normalize_text(entry.get("indicator", ""))
                topic = self._normalize_text(entry.get("topic", ""))
                cat_norm = self._normalize_text(entry.get("category", ""))
                
                # Check for exact structured matches (Direct words)
                for word in direct_words:
                    if word in h_state or word in s_id or word in param or word in ind:
                        score += 5
                        is_direct = True
                    if word in topic:
                        score += 4
                        is_direct = True
                    if word in cat_norm:
                        score += 2
                        is_direct = True
                        
                # Check Keywords array
                kws = entry.get("keywords", [])
                if isinstance(kws, list):
                    for kw in kws:
                        kw_norm = self._normalize_text(str(kw))
                        if direct_words.intersection(kw_norm):
                            score += 3
                            is_direct = True
                        elif related_words.intersection(kw_norm):
                            score += 2
                            is_related = True
                            
                # Check Description matches
                desc_norm = self._normalize_text(entry.get("description", ""))
                desc_match = direct_words.intersection(desc_norm)
                if desc_match:
                    score += len(desc_match) * 1
                    
                rel_desc_match = related_words.intersection(desc_norm)
                if rel_desc_match:
                    score += len(rel_desc_match) * 1
                    is_related = True
                    
                # Other generic fields
                prin_norm = self._normalize_text(entry.get("principle", ""))
                if direct_words.intersection(prin_norm):
                    score += 2
                    is_direct = True
                if related_words.intersection(prin_norm):
                    score += 1
                    is_related = True

                if score > 0 and (is_direct or is_related):
                    kid = entry.get("knowledge_id") or entry.get("id")
                    if kid:
                        if is_direct:
                            has_direct_match = True
                            
                        # Deduplicate by saving highest score
                        if kid not in scored_results or score > scored_results[kid]["score"]:
                            scored_results[kid] = {
                                "source_type": kb_type,
                                "score": score,
                                "priority": priority_map.get(kb_type, 0),
                                "knowledge_id": kid,
                                "content": entry
                            }
                            
        if not scored_results:
            return {"match_status": "NO_MATCH", "results": []}
            
        results_list = list(scored_results.values())
        # Deterministic Ranking: score desc, priority desc, kid asc
        results_list.sort(key=lambda x: (-x["score"], -x["priority"], x["knowledge_id"]))
        
        final_results = results_list[:max_items]
        
        # Clean internal fields
        for r in final_results:
            r.pop("priority", None)
            
        return {
            "match_status": "DIRECT" if has_direct_match else "RELATED",
            "results": final_results
        }

    def get_general_context(self, crop_id: str = "PALAK", max_items: int = 3) -> List[Dict[str, Any]]:
        """Fallback method returning basic crop context."""
        return [
            {
                "source_type": "crop", 
                "score": 1, 
                "knowledge_id": e.get("knowledge_id") or e.get("id"), 
                "content": e
            } 
            for e in self.get_crop_knowledge(crop_id, max_items=max_items)
        ]

    def get_relevant_knowledge(
        self, 
        crop_id: str = "PALAK", 
        stage_id: Optional[str] = None, 
        stage_group: Optional[str] = None, 
        health_state: Optional[str] = None, 
        nutrient_status: Optional[str] = None, 
        sensor_parameters: Optional[List[str]] = None, 
        query: Optional[str] = None, 
        max_items: int = 12
    ) -> Dict[str, Any]:
        """
        Context retrieval for Context Builder combining structural lookups and search.
        """
        scored_results = {}
        priority_map = {"health": 4, "nutrient": 3, "growth": 2, "crop": 1}
        has_direct_match = False
        
        def add_result(kb_type: str, entries: List[Dict], score: int):
            nonlocal has_direct_match
            for e in entries:
                kid = e.get("knowledge_id") or e.get("id")
                if kid:
                    has_direct_match = True # Structured lookups are considered DIRECT matches
                    if kid not in scored_results or score > scored_results[kid]["score"]:
                        scored_results[kid] = {
                            "source_type": kb_type,
                            "score": score,
                            "priority": priority_map.get(kb_type, 0),
                            "knowledge_id": kid,
                            "content": e
                        }

        # STEP 1: Crop Knowledge
        add_result("crop", self.get_crop_knowledge(crop_id, max_items=2), score=10)
        
        # STEP 2: Growth Knowledge
        if stage_id:
            add_result("growth", self.get_growth_knowledge(crop_id, stage_id, max_items=2), score=10)
            if not stage_group:
                stage_group = self.get_stage_group(stage_id)
                
        # STEP 3: Nutrient Knowledge
        if stage_group:
            add_result("nutrient", self.get_nutrient_knowledge(crop_id, stage_group=stage_group, max_items=3), score=10)
            
        if sensor_parameters:
            for p in sensor_parameters:
                add_result("nutrient", self.get_nutrient_knowledge(crop_id, parameter=p, max_items=1), score=10)
                
        if nutrient_status:
            n_status = self._normalize_value(nutrient_status)
            if n_status in self.nutrient_status_map:
                n_status = self.nutrient_status_map[n_status]
            # Simple fallback to search for the status concept
            add_result("nutrient", [res["content"] for res in self.search_knowledge(n_status, crop_id, max_items=1).get("results", [])], score=10)

        # STEP 4: Health Knowledge
        if health_state:
            add_result("health", self.get_health_knowledge(crop_id, health_state=health_state, max_items=2), score=10)
            
        # STEP 5: Query Search
        search_match_status = "NO_MATCH"
        if query:
            search_data = self.search_knowledge(query, crop_id, max_items=5)
            search_match_status = search_data["match_status"]
            for r in search_data["results"]:
                kid = r["knowledge_id"]
                if kid not in scored_results or r["score"] > scored_results[kid]["score"]:
                    r["priority"] = priority_map.get(r["source_type"], 0)
                    scored_results[kid] = r
                    
        # STEP 6-9: Merge, deduplicate, rank and limit
        if not scored_results:
            return {"match_status": "NO_MATCH", "results": []}
            
        results_list = list(scored_results.values())
        results_list.sort(key=lambda x: (-x["score"], -x["priority"], x["knowledge_id"]))
        
        final_results = results_list[:max_items]
        
        for r in final_results:
            r.pop("priority", None)
            
        # Determine overall match status
        final_status = "DIRECT" if has_direct_match or search_match_status == "DIRECT" else (
                       "RELATED" if search_match_status == "RELATED" else "NO_MATCH")
                       
        return {
            "match_status": final_status,
            "results": final_results
        }

if __name__ == "__main__":
    # Test script - strictly testing the loader, not initializing any server/GPT
    current_dir = Path(__file__).parent
    kb_path = current_dir.parent / "kb"
    
    print(f"Loading Knowledge Base from: {kb_path}")
    loader = KBLoader(kb_dir=str(kb_path))
    
    print("\n--- A. Loaded Entries ---")
    for category, entries in loader.kb.items():
        print(f"{category.capitalize()} entries: {len(entries)}")
        
    print("\n--- B. Structured Retrieval ---")
    crop_res = loader.get_crop_knowledge("PALAK")
    print(f"get_crop_knowledge: {len(crop_res)} entries.")
    
    growth_res = loader.get_growth_knowledge("PALAK", "G3")
    print(f"get_growth_knowledge ('G3'): {len(growth_res)} entries.")
    
    nutrient_res = loader.get_nutrient_knowledge("PALAK", stage_group="LATE", parameter="EC")
    print(f"get_nutrient_knowledge ('LATE', 'EC'): {len(nutrient_res)} entries.")
    
    health_res = loader.get_health_knowledge("PALAK", health_state="STRESS")
    print(f"get_health_knowledge ('STRESS'): {len(health_res)} entries.")
    
    print("\n--- C. Direct Search ---")
    direct_search = loader.search_knowledge("yellowing stress pH EC")
    print(f"Status: {direct_search['match_status']}, Results: {len(direct_search['results'])}")
    
    print("\n--- D. Related Search ---")
    related_search = loader.search_knowledge("why does my plant look weak")
    print(f"Status: {related_search['match_status']}, Results: {len(related_search['results'])}")
    for r in related_search["results"][:3]:
        print(f"  ID: {r['knowledge_id']} (Score: {r['score']}, Source: {r['source_type']})")
        
    print("\n--- E. No-match Search ---")
    no_match_search = loader.search_knowledge("market price of spinach")
    print(f"Status: {no_match_search['match_status']}, Results: {len(no_match_search['results'])}")
    
    print("\n--- F. Context Retrieval ---")
    context_res = loader.get_relevant_knowledge(
        crop_id="PALAK",
        stage_id="G3",
        health_state="STRESS",
        query="plant looks weak and yellow"
    )
    print(f"Status: {context_res['match_status']}, Total Results: {len(context_res['results'])}")
    for r in context_res["results"]:
        print(f"  ID: {r['knowledge_id']} (Score: {r['score']}, Source: {r['source_type']})")
