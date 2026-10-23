import React, { useState, useEffect } from 'react';
import { fetchDefaultConfig, createSimulation } from '../api/client';

interface ScenarioConfigProps {
  onSimulationCreated: (simId: string) => void;
}

export const ScenarioConfig: React.FC<ScenarioConfigProps> = ({ onSimulationCreated }) => {
  const [config, setConfig] = useState<any>(null);
  const [seed, setSeed] = useState<number>(42);
  const [policy, setPolicy] = useState<string>('marl');

  useEffect(() => {
    fetchDefaultConfig().then(setConfig).catch(console.error);
  }, []);

  const handlePreset = (type: string) => {
    if (!config) return;
    const newCfg = JSON.parse(JSON.stringify(config));

    if (type === 'earthquake') {
      newCfg.environment.grid_size = [12, 12];
      newCfg.environment.num_zones = 6;
      newCfg.environment.inaccessible_prob = 0.4; // High rubble/blockage
      newCfg.environment.agent_counts = { medical: 3, rescue: 4, logistics: 2 };
    } else if (type === 'flood') {
      newCfg.environment.grid_size = [10, 10];
      newCfg.environment.num_zones = 4;
      newCfg.environment.inaccessible_prob = 0.3;
      newCfg.environment.agent_counts = { medical: 2, rescue: 3, logistics: 3 };
    } else if (type === 'cyclone') {
      newCfg.environment.grid_size = [15, 15];
      newCfg.environment.num_zones = 8;
      newCfg.environment.inaccessible_prob = 0.25;
      newCfg.environment.agent_counts = { medical: 4, rescue: 3, logistics: 4 };
    }

    setConfig(newCfg);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!config) return;
    try {
      const res = await createSimulation({ config, policy, seed });
      onSimulationCreated(res.sim_id);
    } catch (err) {
      console.error(err);
    }
  };

  if (!config) return <div className="card">Loading configuration parameters...</div>;

  const env = config.environment;
  const comms = config.comms || { comm_enabled: true };

  return (
    <div className="card" style={{ maxWidth: 800, margin: '0 auto' }}>
      <div className="card-header">
        <span>⚙️ Scenario Configuration & Presets (FR8)</span>
      </div>

      {/* Presets */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', alignItems: 'center' }}>
        <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Quick Presets:</span>
        <button type="button" className="btn btn-secondary" onClick={() => handlePreset('earthquake')}>🌋 Earthquake (High Damage)</button>
        <button type="button" className="btn btn-secondary" onClick={() => handlePreset('flood')}>🌊 Flood (Medium Hazard)</button>
        <button type="button" className="btn btn-secondary" onClick={() => handlePreset('cyclone')}>🌀 Cyclone (Large Scale)</button>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid-layout-2col">
          {/* Environment Scale */}
          <div>
            <h4 style={{ color: '#38bdf8', marginBottom: '0.75rem' }}>Grid & Time Specs</h4>
            <div className="form-group">
              <label className="form-label">Grid Width / Height</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <input
                  type="number"
                  className="form-input"
                  value={env.grid_size[0]}
                  onChange={(e) => {
                    const val = parseInt(e.target.value) || 10;
                    setConfig({ ...config, environment: { ...env, grid_size: [val, env.grid_size[1]] } });
                  }}
                />
                <input
                  type="number"
                  className="form-input"
                  value={env.grid_size[1]}
                  onChange={(e) => {
                    const val = parseInt(e.target.value) || 10;
                    setConfig({ ...config, environment: { ...env, grid_size: [env.grid_size[0], val] } });
                  }}
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Max Episode Steps</label>
              <input
                type="number"
                className="form-input"
                value={env.max_steps}
                onChange={(e) =>
                  setConfig({ ...config, environment: { ...env, max_steps: parseInt(e.target.value) || 50 } })
                }
              />
            </div>

            <div className="form-group">
              <label className="form-label">Disaster Zones / Depots</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <input
                  type="number"
                  className="form-input"
                  value={env.num_zones}
                  onChange={(e) =>
                    setConfig({ ...config, environment: { ...env, num_zones: parseInt(e.target.value) || 4 } })
                  }
                />
                <input
                  type="number"
                  className="form-input"
                  value={env.num_depots}
                  onChange={(e) =>
                    setConfig({ ...config, environment: { ...env, num_depots: parseInt(e.target.value) || 2 } })
                  }
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Inaccessible Zone Probability</label>
              <input
                type="number"
                step="0.05"
                min="0.0"
                max="0.8"
                className="form-input"
                value={env.inaccessible_prob || 0.2}
                onChange={(e) =>
                  setConfig({ ...config, environment: { ...env, inaccessible_prob: parseFloat(e.target.value) || 0.2 } })
                }
              />
            </div>
          </div>

          {/* Agent Counts & Comms */}
          <div>
            <h4 style={{ color: '#38bdf8', marginBottom: '0.75rem' }}>Agents & Communication</h4>
            <div className="form-group">
              <label className="form-label">Medical Agents</label>
              <input
                type="number"
                className="form-input"
                value={env.agent_counts.medical}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    environment: {
                      ...env,
                      agent_counts: { ...env.agent_counts, medical: parseInt(e.target.value) || 1 }
                    }
                  })
                }
              />
            </div>

            <div className="form-group">
              <label className="form-label">Rescue Agents</label>
              <input
                type="number"
                className="form-input"
                value={env.agent_counts.rescue}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    environment: {
                      ...env,
                      agent_counts: { ...env.agent_counts, rescue: parseInt(e.target.value) || 1 }
                    }
                  })
                }
              />
            </div>

            <div className="form-group">
              <label className="form-label">Logistics Agents</label>
              <input
                type="number"
                className="form-input"
                value={env.agent_counts.logistics}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    environment: {
                      ...env,
                      agent_counts: { ...env.agent_counts, logistics: parseInt(e.target.value) || 1 }
                    }
                  })
                }
              />
            </div>

            <div className="form-group">
              <label className="form-label">Agent Communication Layer</label>
              <select
                className="form-select"
                value={comms.comm_enabled ? 'true' : 'false'}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    comms: { ...comms, comm_enabled: e.target.value === 'true' }
                  })
                }
              >
                <option value="true">Enabled (Messages Broadcast)</option>
                <option value="false">Disabled (No Comms Ablation)</option>
              </select>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem', alignItems: 'center' }}>
          <div style={{ flex: 1 }}>
            <label className="form-label">Initial Seed</label>
            <input
              type="number"
              className="form-input"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
            />
          </div>

          <div style={{ flex: 1 }}>
            <label className="form-label">Policy Selection</label>
            <select className="form-select" value={policy} onChange={(e) => setPolicy(e.target.value)}>
              <option value="marl">MARL (MAPPO)</option>
              <option value="baseline">Greedy Baseline</option>
              <option value="random">Random Policy</option>
            </select>
          </div>

          <div style={{ flex: 1, display: 'flex', alignItems: 'flex-end' }}>
            <button type="submit" className="btn btn-primary" style={{ width: '100%', height: 40, marginTop: 24 }}>
              🚀 Launch Simulation
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};
