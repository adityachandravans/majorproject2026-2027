import { SimulationFrame, CheckpointInfo, RunRecord, EpisodeRecord } from '../types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');

async function apiRequest(path: string, init: RequestInit | undefined, action: string): Promise<Response> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${action} failed (${res.status})${detail ? `: ${detail}` : ''}`);
  }
  return res;
}

export async function fetchDefaultConfig() {
  const res = await apiRequest('/api/config/default', undefined, 'Fetch default config');
  return res.json();
}

export async function createSimulation(payload: {
  config?: any;
  policy?: string;
  checkpoint?: string;
  seed?: number;
  speed?: number;
}): Promise<{ sim_id: string; state: SimulationFrame }> {
  const res = await apiRequest('/api/simulations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }, 'Create simulation');
  return res.json();
}

export async function getSimulationState(sim_id: string): Promise<SimulationFrame> {
  const res = await apiRequest(`/api/simulations/${sim_id}/state`, undefined, 'Get simulation state');
  return res.json();
}

export async function startSimulation(sim_id: string): Promise<SimulationFrame> {
  const res = await apiRequest(`/api/simulations/${sim_id}/start`, { method: 'POST' }, 'Start simulation');
  return res.json();
}

export async function pauseSimulation(sim_id: string): Promise<SimulationFrame> {
  const res = await apiRequest(`/api/simulations/${sim_id}/pause`, { method: 'POST' }, 'Pause simulation');
  return res.json();
}

export async function resetSimulation(sim_id: string): Promise<SimulationFrame> {
  const res = await apiRequest(`/api/simulations/${sim_id}/reset`, { method: 'POST' }, 'Reset simulation');
  return res.json();
}

export async function stepSimulation(sim_id: string): Promise<SimulationFrame> {
  const res = await apiRequest(`/api/simulations/${sim_id}/step`, { method: 'POST' }, 'Step simulation');
  return res.json();
}

export async function switchSimulationPolicy(sim_id: string, policy: string, checkpoint?: string): Promise<SimulationFrame> {
  const res = await apiRequest(`/api/simulations/${sim_id}/policy`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ policy, checkpoint })
  }, 'Switch simulation policy');
  return res.json();
}

export async function fetchCheckpoints(): Promise<CheckpointInfo[]> {
  const res = await apiRequest('/api/checkpoints', undefined, 'Fetch checkpoints');
  return res.json();
}

export async function fetchRuns(): Promise<RunRecord[]> {
  const res = await apiRequest('/api/runs', undefined, 'Fetch runs');
  return res.json();
}

export async function fetchRunEpisodes(run_id: string): Promise<EpisodeRecord[]> {
  const res = await apiRequest(`/api/runs/${run_id}/episodes`, undefined, 'Fetch run episodes');
  return res.json();
}

export async function fetchTrainingCurve(run_id: string): Promise<any[]> {
  const res = await apiRequest(`/api/training/${run_id}/curve`, undefined, 'Fetch training curve');
  return res.json();
}

export async function comparePolicies(payload: { config?: any; seed: number; max_steps?: number }): Promise<any> {
  const res = await apiRequest('/api/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }, 'Compare policies');
  return res.json();
}

export async function fetchResultsSummary(): Promise<{ summary_markdown: string; raw_data: any[] }> {
  const res = await apiRequest('/api/results/summary', undefined, 'Fetch results summary');
  return res.json();
}

export function connectSimulationWebSocket(
  sim_id: string,
  onFrame: (frame: SimulationFrame) => void,
  onError?: (err: Event) => void
): WebSocket {
  const wsUrl = new URL(
    `/ws/simulations/${encodeURIComponent(sim_id)}`,
    API_BASE || window.location.origin
  );
  wsUrl.protocol = wsUrl.protocol === 'https:' ? 'wss:' : 'ws:';
  
  const ws = new WebSocket(wsUrl.toString());

  ws.onmessage = (event) => {
    try {
      const frame: SimulationFrame = JSON.parse(event.data);
      onFrame(frame);
    } catch (e) {
      console.error('Error parsing WS frame', e);
    }
  };

  if (onError) {
    ws.onerror = onError;
  }

  return ws;
}
