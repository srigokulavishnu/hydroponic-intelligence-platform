const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const handleResponse = async (response) => {
  if (!response.ok) {
    let errorMsg = `HTTP Error: ${response.status}`;
    try {
      const errData = await response.json();
      errorMsg = errData.detail || errorMsg;
    } catch (e) {
      // Ignored
    }
    throw new Error(errorMsg);
  }
  return response.json();
};

export const fetchSystemHealth = async () => {
  const res = await fetch(`${API_BASE_URL}/health`);
  return handleResponse(res);
};

export const fetchSystemInfo = async () => {
  const res = await fetch(`${API_BASE_URL}/api/system/info`);
  return handleResponse(res);
};

export const fetchPlantsList = async () => {
  const res = await fetch(`${API_BASE_URL}/api/plants`);
  return handleResponse(res);
};

export const fetchPlantState = async (plantId) => {
  const res = await fetch(`${API_BASE_URL}/api/plants/${plantId}/state`);
  return handleResponse(res);
};

export const fetchPlantHistory = async (plantId, limit = 10) => {
  const res = await fetch(`${API_BASE_URL}/api/plants/${plantId}/history?limit=${limit}`);
  return handleResponse(res);
};

export const sendChatMessage = async (plantId, message, currentState = null, cropId = "PALAK") => {
  const res = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plant_id: plantId, message, crop_id: cropId, current_state: currentState }),
  });
  return handleResponse(res);
};
