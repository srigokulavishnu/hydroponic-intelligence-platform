import { useState, useEffect, useCallback } from "react";
import { fetchSystemInfo, fetchPlantsList, fetchPlantState, fetchPlantHistory, fetchSystemHealth } from "../services/api";

export const usePlantData = () => {
  const [systemInfo, setSystemInfo] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [plantsList, setPlantsList] = useState([]);
  const [selectedPlantId, setSelectedPlantId] = useState("");
  
  const [plantState, setPlantState] = useState(null);
  const [plantHistory, setPlantHistory] = useState([]);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [chatMessages, setChatMessages] = useState([]);

  // Initialize System
  useEffect(() => {
    const initialize = async () => {
      try {
        const [info, health, plantsRes] = await Promise.all([
          fetchSystemInfo(),
          fetchSystemHealth(),
          fetchPlantsList()
        ]);
        
        setSystemInfo(info);
        setSystemHealth(health);
        
        if (plantsRes && plantsRes.plants && plantsRes.plants.length > 0) {
          setPlantsList(plantsRes.plants);
        }
      } catch (err) {
        setError("Failed to initialize system. Please ensure the backend is running.");
      } finally {
        setLoading(false);
      }
    };
    initialize();
  }, []);

  // Fetch Plant Data
  const loadPlantData = useCallback(async (plantId) => {
    if (!plantId) return;
    setLoading(true);
    setError(null);
    try {
      const [stateRes, historyRes] = await Promise.all([
        fetchPlantState(plantId),
        fetchPlantHistory(plantId)
      ]);
      setPlantState(stateRes);
      setPlantHistory(historyRes.history || []);
    } catch (err) {
      setError(`Unable to load data for ${plantId}: ${err.message}`);
      setPlantState(null);
      setPlantHistory([]);
    } finally {
      setLoading(false);
    }
  }, []);

  // Trigger fetch on plant change
  useEffect(() => {
    if (selectedPlantId) {
      setChatMessages([]); // Clear chat history on plant change
      loadPlantData(selectedPlantId);
    }
  }, [selectedPlantId, loadPlantData]);

  const refreshData = () => {
    if (selectedPlantId) loadPlantData(selectedPlantId);
  };

  return {
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
  };
};
