export interface AgentState {
  pos: [number, number];
  role: 'medical' | 'rescue' | 'logistics';
  status: number;
  target_zone: number;
  carrying: number;
  supply_payload: number;
  has_ambulance: boolean;
  has_rescue_team: boolean;
}

export interface ZoneState {
  id: number;
  pos: [number, number];
  severity: number;
  casualty_count: number;
  trapped_count: number;
  casualties_treated: number;
  casualties_lost: number;
  accessibility: boolean;
  supply_demand: number;
  supplies_delivered: number;
}

export interface DepotState {
  id: number;
  pos: [number, number];
  ambulances: number;
  rescue_teams: number;
  supply_units: number;
  max_ambulances: number;
  max_rescue_teams: number;
  max_supply_units: number;
}

export interface SimulationMetrics {
  avg_response_time: number;
  resource_utilization_rate: number;
  task_completion_rate: number;
  coordination_efficiency: number;
  casualties_served: number;
  casualties_lost: number;
  casualties_total: number;
  total_reward?: number;
}

export interface SimulationFrame {
  sim_id: string;
  grid_size: [number, number];
  step_count: number;
  max_steps: number;
  done: boolean;
  policy: string;
  status: 'created' | 'running' | 'paused' | 'completed';
  last_step_latency_ms: number;
  zones: ZoneState[];
  depots: DepotState[];
  agents: Record<string, AgentState>;
  metrics: SimulationMetrics;
  actions?: Record<string, number>;
  rewards?: Record<string, number>;
}

export interface CheckpointInfo {
  filename: string;
  path: string;
  size_bytes: number;
  modified_time: string;
}

export interface RunRecord {
  run_id: string;
  run_type: string;
  policy: string;
  timestamp: string;
  notes: string;
}

export interface EpisodeRecord {
  episode_id: number;
  run_id: string;
  episode_num: number;
  total_reward: number;
  avg_response_time: number;
  resource_utilization_rate: number;
  casualties_served: number;
  casualties_lost: number;
  casualties_total: number;
  task_completion_rate: number;
  coordination_efficiency: number;
  medical_reward: number;
  rescue_reward: number;
  logistics_reward: number;
}
