import { SimulationFrame, CheckpointInfo, RunRecord, EpisodeRecord } from '../types';

const API_BASE = '';

export async function fetchDefaultConfig() {
  const res = await fetch(`${API_BASE}/api/config/default`);
  if (!res.ok) throw new Error('Failed to fetch default config');
  return res.json();
}

export async function createSimulation(payload: {
  config?: any;
  policy?: string;
  checkpoint?: string;
  seed?: number;
  speed?: number;
}): Promise<{ sim_id: string; state: SimulationFrame }> {
  const res = await fetch(`${API_BASE}/api/simulations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error('Failed to create simulation');
  return res.json();
}

export async function startSimulation(sim_id: string): Promise<SimulationFrame> {
  const res = await fetch(`${API_BASE}/api/simulations/${sim_id}/start`, { method: 'POST' });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Failed to start simulation (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function pauseSimulation(sim_id: string): Promise<SimulationFrame> {
  const res = await fetch(`${API_BASE}/api/simulations/${sim_id}/pause`, { method: 'POST' });
  return res.json();
}

export async function resetSimulation(sim_id: string): Promise<SimulationFrame> {
  const res = await fetch(`${API_BASE}/api/simulations/${sim_id}/reset`, { method: 'POST' });
  return res.json();
}

export async function stepSimulation(sim_id: string): Promise<SimulationFrame> {
  const res = await fetch(`${API_BASE}/api/simulations/${sim_id}/step`, { method: 'POST' });
  return res.json();
}

export async function switchSimulationPolicy(sim_id: string, policy: string, checkpoint?: string): Promise<SimulationFrame> {
  const res = await fetch(`${API_BASE}/api/simulations/${sim_id}/policy`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ policy, checkpoint })
  });
  return res.json();
}

export async function fetchCheckpoints(): Promise<CheckpointInfo[]> {
  const res = await fetch(`${API_BASE}/api/checkpoints`);
  return res.json();
}

export async function fetchRuns(): Promise<RunRecord[]> {
  const res = await fetch(`${API_BASE}/api/runs`);
  return res.json();
}

export async function fetchRunEpisodes(run_id: string): Promise<EpisodeRecord[]> {
  const res = await fetch(`${API_BASE}/api/runs/${run_id}/episodes`);
  return res.json();
}

export async function fetchTrainingCurve(run_id: string): Promise<any[]> {
  const res = await fetch(`${API_BASE}/api/training/${run_id}/curve`);
  return res.json();
}

export async function comparePolicies(payload: { config?: any; seed: number; max_steps?: number }): Promise<any> {
  const res = await fetch(`${API_BASE}/api/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  return res.json();
}

export async function fetchResultsSummary(): Promise<{ summary_markdown: string; raw_data: any[] }> {
  const res = await fetch(`${API_BASE}/api/results/summary`);
  return res.json();
}

export function connectSimulationWebSocket(
  sim_id: string,
  onFrame: (frame: SimulationFrame) => void,
  onError?: (err: Event) => void
): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const wsUrl = `${protocol}//${host}/ws/simulations/${sim_id}`;
  
  const ws = new WebSocket(wsUrl);

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
