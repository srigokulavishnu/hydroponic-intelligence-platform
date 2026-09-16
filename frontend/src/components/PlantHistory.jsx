import React, { useState } from "react";
import { format } from "date-fns";
import { ChevronDown, ChevronUp } from "lucide-react";

export const PlantHistory = ({ history, loading }) => {
  const [expanded, setExpanded] = useState(false);

  if (loading) {
    return (
      <div className="card animate-fade-in">
        <h4 className="text-sm font-semibold" style={{ marginBottom: "1rem" }}>Recent History</h4>
        <div className="text-sm text-muted">Loading history...</div>
      </div>
    );
  }

  if (!history || history.length === 0) {
    return (
      <div className="card animate-fade-in">
        <h4 className="text-sm font-semibold" style={{ marginBottom: "1rem" }}>Recent History</h4>
        <div className="text-sm text-muted">No historical observations yet.</div>
      </div>
    );
  }

  const displayHistory = expanded ? history : history.slice(0, 3);

  const getHealthColor = (h) => (h === "OPTIMAL" || h === "OK") ? "text-ok" : "text-error";

  return (
    <div className="card animate-fade-in">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <h4 className="text-sm font-semibold m-0">Recent History</h4>
        {history.length > 3 && (
          <button 
            className="btn" 
            style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem" }} 
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? (
              <><ChevronUp size={14} /> Show Less</>
            ) : (
              <><ChevronDown size={14} /> Show More ({history.length})</>
            )}
          </button>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
        {displayHistory.map((entry, idx) => (
          <div key={idx} style={{ 
            padding: "0.75rem", 
            border: "1px solid var(--border-subtle)", 
            borderRadius: "var(--radius-sm)",
            background: "var(--bg-app)",
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: "0.5rem",
            alignItems: "center"
          }}>
            <div>
              <div className="text-xs text-muted">
                {entry.timestamp ? format(new Date(entry.timestamp), "MMM d, HH:mm") : "Unknown time"}
              </div>
              <div className="text-sm font-medium">
                pH: {entry.ph || "--"} | EC: {entry.ec || "--"}
              </div>
            </div>
            
            <div className="text-sm">
              <div className={getHealthColor(entry.health)}>{entry.health || "Health: --"}</div>
              <div className="text-muted text-xs">{entry.growth || "Growth: --"}</div>
            </div>

            <div className="text-sm text-right">
              {entry.action && <div className="text-accent font-medium text-xs">{entry.action}</div>}
              {entry.outcome && <div className="text-muted text-xs">{entry.outcome}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
