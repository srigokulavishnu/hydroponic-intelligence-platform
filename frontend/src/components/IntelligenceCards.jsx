import React from "react";

const ConfidenceScore = ({ confidence }) => {
  if (confidence == null) return null;
  const percentage = Math.round(confidence * 100);
  return <span className="text-xs text-muted">{percentage}% model confidence</span>;
};

export const IntelligenceCards = ({ predictions, loading }) => {
  if (loading) {
    return (
      <div className="intelligence-grid animate-fade-in">
        {[1, 2, 3].map(i => (
          <div key={i} className="metric-tile" style={{ opacity: 0.5 }}>
            <div style={{ height: "14px", width: "40%", background: "var(--border-subtle)", borderRadius: "4px", marginBottom: "8px" }}></div>
            <div style={{ height: "20px", width: "60%", background: "var(--border-strong)", borderRadius: "4px" }}></div>
          </div>
        ))}
      </div>
    );
  }

  const getStatusColor = (status, label) => {
    if (status === "UNAVAILABLE") return "unavailable";
    if (status === "ERROR") return "error";
    if (label === "STRESS" || label === "NON_OPTIMAL") return "error";
    if (label === "OPTIMAL") return "ok";
    return "text-primary"; // Default
  };

  const health = predictions?.health || { status: "UNAVAILABLE" };
  const growth = predictions?.growth || { status: "UNAVAILABLE" };
  const nutrient = predictions?.nutrient || { status: "UNAVAILABLE" };

  return (
    <div className="intelligence-grid animate-fade-in">
      <div className="metric-tile">
        <span className="text-xs text-secondary font-medium" style={{ textTransform: "uppercase" }}>Health</span>
        <span className={`font-semibold text-${getStatusColor(health.status, health.label)}`} style={{ fontSize: "1.125rem", margin: "0.25rem 0" }}>
          {health.status === "OK" ? health.label : "Not available"}
        </span>
        {health.status === "OK" ? (
          <ConfidenceScore confidence={health.confidence} />
        ) : (
          <span className="text-xs text-muted">{health.reason || "Status unknown"}</span>
        )}
      </div>

      <div className="metric-tile">
        <span className="text-xs text-secondary font-medium" style={{ textTransform: "uppercase" }}>Growth</span>
        <span className={`font-semibold text-${getStatusColor(growth.status, growth.label)}`} style={{ fontSize: "1.125rem", margin: "0.25rem 0" }}>
          {growth.status === "OK" ? growth.label : "Not available"}
        </span>
        {growth.status === "OK" ? (
          <ConfidenceScore confidence={growth.confidence} />
        ) : (
          <span className="text-xs text-muted">Growth model not trained yet.</span>
        )}
      </div>

      <div className="metric-tile">
        <span className="text-xs text-secondary font-medium" style={{ textTransform: "uppercase" }}>Nutrient</span>
        <span className={`font-semibold text-${getStatusColor(nutrient.status, nutrient.label)}`} style={{ fontSize: "1.125rem", margin: "0.25rem 0" }}>
          {nutrient.status === "OK" ? nutrient.label : "Not available"}
        </span>
        {nutrient.status === "OK" ? (
          <ConfidenceScore confidence={nutrient.confidence} />
        ) : (
          <span className="text-xs text-muted">{nutrient.reason || "Status unknown"}</span>
        )}
      </div>
    </div>
  );
};
