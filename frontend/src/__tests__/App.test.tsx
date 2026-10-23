import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { render, screen } from '@testing-library/react';
import { GridView } from '../components/GridView';
import { Navbar } from '../components/Navbar';
import { SimulationFrame } from '../types';

describe('Frontend React Components', () => {
  it('Navbar renders brand and tabs', () => {
    const handleTab = vi.fn();
    render(<Navbar activeTab="sim" setActiveTab={handleTab} />);
    
    expect(screen.getByText(/MARL Disaster Response System/i)).toBeDefined();
    expect(screen.getByText(/Live Simulation/i)).toBeDefined();
    expect(screen.getByText(/Scenario Config/i)).toBeDefined();
  });

  it('GridView renders agents and zones from simulation frame', () => {
    const dummyFrame: SimulationFrame = {
      sim_id: 'test_sim',
      grid_size: [5, 5],
      step_count: 3,
      max_steps: 15,
      done: false,
      policy: 'marl',
      status: 'running',
      last_step_latency_ms: 12.5,
      zones: [
        {
          id: 0,
          pos: [1, 1],
          severity: 2,
          casualty_count: 3,
          trapped_count: 1,
          casualties_treated: 0,
          casualties_lost: 0,
          accessibility: true,
          supply_demand: 4.0,
          supplies_delivered: 0
        }
      ],
      depots: [
        {
          id: 0,
          pos: [0, 0],
          ambulances: 5,
          rescue_teams: 5,
          supply_units: 20,
          max_ambulances: 5,
          max_rescue_teams: 5,
          max_supply_units: 20
        }
      ],
      agents: {
        medical_0: {
          pos: [0, 0],
          role: 'medical',
          status: 0,
          target_zone: 0,
          carrying: 0,
          supply_payload: 0,
          has_ambulance: true,
          has_rescue_team: false
        }
      },
      metrics: {
        avg_response_time: 3.0,
        resource_utilization_rate: 0.8,
        task_completion_rate: 0.5,
        coordination_efficiency: 1.0,
        casualties_served: 0,
        casualties_lost: 0,
        casualties_total: 4
      }
    };

    const { container } = render(<GridView frame={dummyFrame} />);
    expect(container.querySelector('svg')).not.toBeNull();
    expect(screen.getByText(/Z0 \(S2\)/i)).toBeDefined();
    expect(screen.getByText(/Depot 0/i)).toBeDefined();
  });
});
