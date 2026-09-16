import React from "react";
import { CameraOff, Image as ImageIcon } from "lucide-react";

export const PlantOverview = ({ plantId, plantState, loading }) => {
  // If the backend doesn't provide an image URL, we show a clean unavailable state.
  const imageUrl = plantState?.current_observation?.image_url || null;
  const imageStatus = plantState?.current_observation?.image_status || "Unknown";

  return (
    <div className="card animate-fade-in">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
        <div>
          <h3>{plantId || "Select a plant"}</h3>
          <p className="text-sm">Palak plant observation</p>
        </div>
        {plantState && (
          <div style={{ textAlign: "right" }}>
            <span className="text-xs text-muted" style={{ display: "block", marginBottom: "0.25rem" }}>Overall Status</span>
            <span className={`text-sm font-semibold text-${plantState.overall_status === "OK" ? "ok" : plantState.overall_status === "PARTIAL" ? "warning" : "error"}`}>
              {plantState.overall_status}
            </span>
          </div>
        )}
      </div>

      <div 
        style={{ 
          width: "100%", 
          aspectRatio: "16/9", 
          backgroundColor: "var(--bg-surface-hover)", 
          borderRadius: "var(--radius-md)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          border: "1px dashed var(--border-strong)",
          overflow: "hidden",
          position: "relative"
        }}
      >
        {loading ? (
          <div className="text-muted" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.5rem" }}>
            <div className="status-dot warning" style={{ animation: "fadeIn 1s infinite alternate" }}></div>
            <span className="text-sm">Loading image stream...</span>
          </div>
        ) : imageUrl ? (
          <img src={imageUrl} alt={`Crop of ${plantId}`} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        ) : (
          <div className="text-muted" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.5rem" }}>
            {imageStatus === "OK" ? <ImageIcon size={32} /> : <CameraOff size={32} />}
            <span className="text-sm font-medium">Image unavailable via API</span>
            <span className="text-xs">Backend camera status: {imageStatus}</span>
          </div>
        )}
      </div>
    </div>
  );
};
