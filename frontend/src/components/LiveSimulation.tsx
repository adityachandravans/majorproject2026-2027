import React, { useState, useEffect, useRef } from 'react';
import { GridView } from './GridView';
import { SimulationFrame, AgentState } from '../types';
import { createSimulation, startSimulation, pauseSimulation, resetSimulation, stepSimulation, switchSimulationPolicy, connectSimulationWebSocket } from '../api/client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export const LiveSimulation: React.FC = () => {
  const [frame, setFrame] = useState<SimulationFrame | null>(null);
  const [policy, setPolicy] = useState<string>('marl');
  const [seed, setSeed] = useState<number>(42);
  const [speed, setSpeed] = useState<number>(2.0);
  const [history, setHistory] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Initialize simulation
  const initSim = async (newPolicy = policy, newSeed = seed) => {
    try {
      if (wsRef.current) {
        wsRef.current.close();
      }
      const res = await createSimulation({ policy: newPolicy, seed: newSeed, speed });
      setFrame(res.state);
      setHistory([{ step: 0, ...res.state.metrics, reward: 0 }]);

      // Connect WebSocket
      const ws = connectSimulationWebSocket(res.sim_id, (newFrame) => {
        setFrame(newFrame);
        setHistory((prev) => {
          const newPoint = {
            step: newFrame.step_count,
            ...newFrame.metrics,
            reward: newFrame.metrics.total_reward || 0
          };
          if (prev.length > 0 && prev[prev.length - 1].step === newFrame.step_count) {
            return prev;
          }
          return [...prev, newPoint];
        });
      });
      wsRef.current = ws;
    } catch (e) {
      console.error('Failed to init simulation', e);
    }
  };

  useEffect(() => {
    initSim();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const handleStart = async () => {
    if (!frame) return;
    setError(null);
    try {
      await startSimulation(frame.sim_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const handlePause = async () => {
    if (!frame) return;
    await pauseSimulation(frame.sim_id);
  };

  const handleStep = async () => {
    if (!frame) return;
    const newFrame = await stepSimulation(frame.sim_id);
    setFrame(newFrame);
  };

  const handleReset = async () => {
    if (!frame) return;
    const newFrame = await resetSimulation(frame.sim_id);
    setFrame(newFrame);
    setHistory([]);
  };

  const handlePolicyChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newPol = e.target.value;
    setPolicy(newPol);
    if (frame) {
      const updatedFrame = await switchSimulationPolicy(frame.sim_id, newPol);
      setFrame(updatedFrame);
    }
  };

  return (
    <div>
      {error && <div role="alert">{error}</div>}
      {/* 1. Control Bar */}
      <div className="controls-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label style={{ fontSize: '0.9rem', fontWeight: 600 }}>Policy:</label>
          <select value={policy} onChange={handlePolicyChange} className="form-select" style={{ width: 140 }}>
            <option value="marl">MARL (MAPPO)</option>
            <option value="baseline">Greedy Baseline</option>
            <option value="random">Random Policy</option>
          </select>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button className="btn btn-primary" onClick={handleStart} disabled={frame?.status === 'running'}>▶ Start</button>
          <button className="btn btn-warning" onClick={handlePause} disabled={frame?.status !== 'running'}>⏸ Pause</button>
          <button className="btn btn-secondary" onClick={handleStep}>⏭ Step</button>
          <button className="btn btn-secondary" onClick={handleReset}>🔄 Reset</button>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label style={{ fontSize: '0.9rem' }}>Speed: {speed}x</label>
          <input
            type="range"
            min="0.5"
            max="5.0"
            step="0.5"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            style={{ width: 100 }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label style={{ fontSize: '0.9rem' }}>Seed:</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
            className="form-input"
            style={{ width: 80 }}
          />
          <button className="btn btn-secondary" onClick={() => initSim(policy, seed)}>New Scenario</button>
        </div>

        {frame && (
          <div style={{ marginLeft: 'auto', fontSize: '0.85rem', color: '#94a3b8' }}>
            Step Latency: <strong style={{ color: '#38bdf8' }}>{frame.last_step_latency_ms} ms</strong>
          </div>
        )}
      </div>

      {/* 2. Main Grid and Agent Status */}
      <div className="grid-layout-2col">
        <GridView frame={frame} />

        <div className="card">
          <div className="card-header">
            <span>🤖 Agent Status & Resources</span>
            <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>{frame ? Object.keys(frame.agents).length : 0} Active Agents</span>
          </div>

          <table className="data-table">
            <thead>
              <tr>
                <th>Agent</th>
                <th>Role</th>
                <th>Pos</th>
                <th>Resource / Status</th>
                <th>Target</th>
              </tr>
            </thead>
            <tbody>
              {frame &&
                Object.entries(frame.agents).map(([aId, aData]: [string, AgentState]) => (
                  <tr key={aId}>
                    <td style={{ fontWeight: 600 }}>{aId}</td>
                    <td>
                      <span
                        style={{
                          padding: '0.2rem 0.5rem',
                          borderRadius: 4,
                          fontSize: '0.75rem',
                          fontWeight: 700,
                          backgroundColor:
                            aData.role === 'medical' ? '#7f1d1d' : aData.role === 'rescue' ? '#78350f' : '#064e3b',
                          color:
                            aData.role === 'medical' ? '#fca5a5' : aData.role === 'rescue' ? '#fde68a' : '#a7f3d0'
                        }}
                      >
                        {aData.role.toUpperCase()}
                      </span>
                    </td>
                    <td>({aData.pos[0]}, {aData.pos[1]})</td>
                    <td>
                      {aData.has_ambulance && '🚑 Ambulance '}
                      {aData.has_rescue_team && '🛠️ Rescue Team '}
                      {aData.supply_payload > 0 && `📦 Payload: ${aData.supply_payload} `}
                      {!aData.has_ambulance && !aData.has_rescue_team && aData.supply_payload === 0 && 'Idle / Searching'}
                    </td>
                    <td>{aData.target_zone >= 0 ? `Zone ${aData.target_zone}` : 'None'}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. Live Charts */}
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div className="card-header">
          <span>📈 Real-Time Performance Metrics</span>
        </div>

        <div style={{ height: 300, width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="step" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} />
              <Legend />
              <Line type="monotone" dataKey="task_completion_rate" name="Task Completion" stroke="#3b82f6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="coordination_efficiency" name="Coordination Eff." stroke="#10b981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="resource_utilization_rate" name="Resource Util." stroke="#f59e0b" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="avg_response_time" name="Avg Response Time" stroke="#ef4444" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
