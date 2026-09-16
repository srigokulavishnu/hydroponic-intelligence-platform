import React from "react";
import { Sprout } from "lucide-react";
import { SystemStatus } from "./SystemStatus";

export const Header = ({ systemInfo, systemHealth }) => {
  const getStatusColor = (health) => {
    if (!health || health.status === "ERROR") return "error";
    if (health.status === "PARTIAL") return "warning";
    return "ok";
  };

  const statusColor = getStatusColor(systemHealth);
  const statusText = systemHealth ? `System ${systemHealth.status}` : "Connecting...";

  return (
    <header className="header">
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <Sprout size={28} className="text-accent" />
        <div>
          <h2 style={{ marginBottom: "0.25rem" }}>
            {systemInfo?.project || "Adaptive Hydroponic Intelligence"}
          </h2>
          <div className="text-sm text-muted">
            {systemInfo?.system || "NFT"} &bull; {systemInfo?.crop || "PALAK"} &bull; Section {systemInfo?.section || "C01"}
          </div>
        </div>
      </div>
      
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }} className="text-sm font-medium">
          <span className={`status-dot ${statusColor}`}></span>
          {statusText}
        </div>
        <div style={{ display: "flex", alignItems: "center" }}>
          <SystemStatus health={systemHealth} />
        </div>
      </div>
    </header>
  );
};
