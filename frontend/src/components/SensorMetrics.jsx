import React from "react";
import { format } from "date-fns";

export const SensorMetrics = ({ currentObservation, loading }) => {
  if (loading) {
    return (
      <div className="card animate-fade-in">
        <h4 className="text-sm font-semibold" style={{ marginBottom: "1rem" }}>Current Environment</h4>
        <div className="sensor-grid">
          {[1, 2, 3, 4, 5].map(i => (
            <div key={i} className="metric-tile" style={{ opacity: 0.5, height: "60px" }}></div>
          ))}
        </div>
      </div>
    );
  }

  const sensors = currentObservation?.sensors;
  const timestamp = currentObservation?.timestamp;

  const SensorTile = ({ label, value, unit }) => (
    <div className="metric-tile">
      <span className="text-xs text-secondary font-medium">{label}</span>
      <span className="font-semibold text-primary" style={{ fontSize: "1.125rem", marginTop: "0.25rem" }}>
        {value !== null && value !== undefined ? (
          <>
            {value} <span className="text-xs text-muted font-normal">{unit}</span>
          </>
        ) : (
          <span className="text-sm text-muted">--</span>
        )}
      </span>
    </div>
  );

  return (
    <div className="card animate-fade-in">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <h4 className="text-sm font-semibold m-0">Current Environment</h4>
        {timestamp && (
          <span className="text-xs text-muted">
            Updated: {format(new Date(timestamp), "HH:mm:ss")}
          </span>
        )}
      </div>

      {!sensors ? (
        <div className="text-sm text-muted">Current sensor readings are temporarily unavailable.</div>
      ) : (
        <div className="sensor-grid">
          <SensorTile label="pH" value={sensors.ph} unit="" />
          <SensorTile label="EC" value={sensors.ec?.toFixed(2)} unit="mS/cm" />
          <SensorTile label="Temp" value={sensors.temperature} unit="°C" />
          <SensorTile label="Humidity" value={sensors.humidity} unit="%" />
          <SensorTile label="Light" value={sensors.light} unit="lx" />
        </div>
      )}
    </div>
  );
};
