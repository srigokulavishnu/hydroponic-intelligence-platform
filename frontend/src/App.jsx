import React from "react";
import { usePlantData } from "./hooks/usePlantData";
import { Header } from "./components/Header";
import { PlantSelector } from "./components/PlantSelector";
import { PlantOverview } from "./components/PlantOverview";
import { IntelligenceCards } from "./components/IntelligenceCards";
import { SensorMetrics } from "./components/SensorMetrics";
import { PlantHistory } from "./components/PlantHistory";
import { AssistantPanel } from "./components/AssistantPanel";
import { SystemStatus } from "./components/SystemStatus";

function App() {
  const {
    systemInfo,
    systemHealth,
    plantsList,
    selectedPlantId,
    setSelectedPlantId,
    plantState,
    plantHistory,
    loading,
    error,
    refreshData,
    chatMessages,
    setChatMessages
  } = usePlantData();

  // If initial load fails catastrophically
  if (error && !systemInfo) {
    return (
      <div style={{ height: "100vh", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: "1rem" }}>
        <h2 className="text-error">Initialization Failed</h2>
        <p className="text-muted">{error}</p>
        <button className="btn btn-primary" onClick={() => window.location.reload()}>Retry Connection</button>
      </div>
    );
  }

  return (
    <>
      <div style={{ position: "absolute", top: "1rem", right: "1.5rem", zIndex: 10 }}>
        <SystemStatus health={systemHealth} />
      </div>

      <Header systemInfo={systemInfo} systemHealth={systemHealth} />

      <main className="app-container">
        <div className="main-layout">
          
          {/* LEFT SIDE: Overview, Metrics, History */}
          <div style={{ display: "flex", flexDirection: "column" }}>
            <PlantSelector 
              plantsList={plantsList} 
              selectedPlantId={selectedPlantId} 
              onSelect={setSelectedPlantId}
              loading={loading}
              onRefresh={refreshData}
            />

            {error && systemInfo && (
              <div className="card" style={{ borderLeft: "4px solid var(--status-error)", color: "var(--status-error)" }}>
                <p className="text-sm font-medium">{error}</p>
              </div>
            )}

            {!selectedPlantId ? (
              <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "400px", borderStyle: "dashed" }}>
                <div style={{ padding: "1.5rem", borderRadius: "50%", background: "rgba(255,255,255,0.05)", marginBottom: "1.5rem" }}>
                  <span style={{ fontSize: "2rem" }}>🌱</span>
                </div>
                <h2 style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>Plant Overview</h2>
                <p className="text-secondary" style={{ marginBottom: "1rem" }}>Select a plant to view its current state.</p>
                <p className="text-sm text-muted text-center" style={{ maxWidth: "350px" }}>
                  Choose a plant to view its current image, sensor state, AI predictions, and history.
                </p>
              </div>
            ) : (
              <>
                <PlantOverview 
                  plantId={selectedPlantId} 
                  plantState={plantState} 
                  loading={loading} 
                />

                <IntelligenceCards 
                  predictions={plantState?.predictions} 
                  loading={loading} 
                />

                <SensorMetrics 
                  currentObservation={plantState?.current_observation} 
                  loading={loading} 
                />

                <PlantHistory 
                  history={plantHistory} 
                  loading={loading} 
                />
              </>
            )}
          </div>

          {/* RIGHT SIDE: AI Assistant Panel */}
          <div>
            <AssistantPanel 
              plantId={selectedPlantId}
              plantState={plantState}
              messages={chatMessages}
              setMessages={setChatMessages}
            />
          </div>

        </div>
      </main>
    </>
  );
}

export default App;
