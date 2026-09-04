/**
 * RiskGuard AI — API Client
 * Fetch wrapper for communicating with the FastAPI backend.
 */

const API_BASE = '/api';

async function fetchJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }
  return res.json();
}

export const api = {
  getDashboardMetrics: () => fetchJson('/dashboard/metrics'),
  getRiskTrend: () => fetchJson('/dashboard/risk-trend'),
  getIncidents: () => fetchJson('/incidents'),
  getIncidentDetail: (id) => fetchJson(`/incidents/${id}`),
  getThresholdLab: () => fetchJson('/threshold-lab'),
  getCostOfInaction: () => fetchJson('/benchmarks/cost-of-inaction'),
  getAdversarial: () => fetchJson('/benchmarks/adversarial'),
  getDrift: () => fetchJson('/benchmarks/drift'),
  getLatency: () => fetchJson('/benchmarks/latency'),
  getFlashSale: () => fetchJson('/benchmarks/flash-sale'),
  getEvaluationMatrix: () => fetchJson('/evaluation-matrix'),
  triggerInvestigation: async (id) => {
    const res = await fetch(`${API_BASE}/incidents/${id}/investigate`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `API error: ${res.status}`);
    }
    return res.json();
  },
};
