import React from "react";
import { Search } from "lucide-react";

export const PlantSelector = ({ plantsList, selectedPlantId, onSelect, loading, onRefresh }) => {
  return (
    <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <div style={{ position: "relative" }}>
          <Search size={16} className="text-muted" style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)" }} />
          <select 
            className="input" 
            style={{ paddingLeft: "2.25rem", width: "200px", appearance: "none", cursor: "pointer", fontWeight: "500" }}
            value={selectedPlantId || ""}
            onChange={(e) => onSelect(e.target.value)}
            disabled={loading}
          >
            <option value="">[ Select a plant... ]</option>
            {plantsList.map(p => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
          <div style={{ position: "absolute", right: "12px", top: "50%", transform: "translateY(-50%)", pointerEvents: "none", color: "var(--text-muted)", fontSize: "12px" }}>▼</div>
        </div>
        <span className="text-sm text-secondary">
          {plantsList.length > 0 ? "Select a plant to view its current state" : "No plants available"}
        </span>
      </div>

      {selectedPlantId && (
        <button className="btn" onClick={onRefresh} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh State"}
        </button>
      )}
    </div>
  );
};
