import React, { useState } from "react";
import { createPortal } from "react-dom";
import { Activity, X } from "lucide-react";

export const SystemStatus = ({ health }) => {
  const [open, setOpen] = useState(false);

  if (!health) return null;

  const StatusItem = ({ label, statusStr, subtext }) => {
    let color = "ok";
    if (statusStr === "ERROR" || statusStr === "UNAVAILABLE") color = "error";
    
    // Growth model explicitly gets a pass as unavailable since it's untrained
    if (label === "Growth" && statusStr === "UNAVAILABLE") color = "unavailable";

    return (
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.5rem 0", borderBottom: "1px solid var(--border-subtle)" }}>
        <div>
          <div className="text-sm font-medium">{label}</div>
          {subtext && <div className="text-xs text-muted">{subtext}</div>}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span className={`status-dot ${color}`}></span>
          <span className="text-xs font-medium" style={{ textTransform: "capitalize" }}>
            {statusStr === "OK" ? "Online" : statusStr.toLowerCase()}
          </span>
        </div>
      </div>
    );
  };

  return (
    <>
      <button className="btn" onClick={() => setOpen(true)} style={{ padding: "0.5rem" }}>
        <Activity size={18} className="text-secondary" />
      </button>

      {open && createPortal(
        <div style={{ 
          position: "fixed", top: 0, left: 0, right: 0, bottom: 0, 
          background: "rgba(0,0,0,0.5)", zIndex: 1000,
          display: "flex", alignItems: "center", justifyContent: "center" 
        }}>
          <div className="card animate-fade-in" style={{ width: "100%", maxWidth: "400px", margin: "1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <h3 style={{ margin: 0 }}>System Status</h3>
              <button onClick={() => setOpen(false)} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)" }}>
                <X size={20} />
              </button>
            </div>

            <div style={{ display: "flex", flexDirection: "column" }}>
              <StatusItem label="Health Model" statusStr={health.components?.model_loader?.health?.status || "ERROR"} subtext="ConvNeXt-Tiny" />
              <StatusItem label="Growth Model" statusStr={health.components?.model_loader?.growth?.status || "UNAVAILABLE"} subtext="Not trained yet" />
              <StatusItem label="Nutrient Model" statusStr={health.components?.model_loader?.nutrient?.status || "ERROR"} subtext="XGBoost" />
              <StatusItem label="Knowledge Base" statusStr={health.components?.knowledge_base || "ERROR"} />
              <StatusItem label="History Store" statusStr={health.components?.history || "ERROR"} />
              <StatusItem label="Gemini AI" statusStr={health.components?.llm?.status || "ERROR"} />
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
};
