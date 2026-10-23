import React, { useState } from 'react';
import { GridView } from './GridView';
import { comparePolicies } from '../api/client';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export const ComparePolicies: React.FC = () => {
  const [seed, setSeed] = useState<number>(42);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<any>(null);

  const handleRunComparison = async () => {
    setLoading(true);
    try {
      const res = await comparePolicies({ seed, max_steps: 40 });
      setResult(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const chartData = result
    ? [
        {
          metric: 'Task Completion',
          MARL: (result.marl_metrics.task_completion_rate * 100).toFixed(1),
          Baseline: (result.baseline_metrics.task_completion_rate * 100).toFixed(1)
        },
        {
          metric: 'Coordination Eff.',
          MARL: (result.marl_metrics.coordination_efficiency * 100).toFixed(1),
          Baseline: (result.baseline_metrics.coordination_efficiency * 100).toFixed(1)
        },
        {
          metric: 'Resource Util.',
          MARL: (result.marl_metrics.resource_utilization_rate * 100).toFixed(1),
          Baseline: (result.baseline_metrics.resource_utilization_rate * 100).toFixed(1)
        }
      ]
    : [];

  return (
    <div>
      <div className="controls-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label style={{ fontWeight: 600 }}>Comparison Seed:</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
            className="form-input"
            style={{ width: 100 }}
          />
        </div>

        <button className="btn btn-primary" onClick={handleRunComparison} disabled={loading}>
          {loading ? '⏳ Running Benchmark...' : '⚔️ Run MARL vs Baseline Comparison'}
        </button>
      </div>

      {result && (
        <>
          {/* Side-by-Side Grid Maps */}
          <div className="grid-layout-2col">
            <div className="card">
              <GridView frame={result.marl_history[result.marl_history.length - 1]} title="🤖 MARL (MAPPO Policy)" />
            </div>

            <div className="card">
              <GridView frame={result.baseline_history[result.baseline_history.length - 1]} title="📋 Greedy Baseline Policy" />
            </div>
          </div>

          {/* Metric Comparison Table & Winner Highlights */}
          <div className="grid-layout-2col" style={{ marginTop: '1.5rem' }}>
            <div className="card">
              <div className="card-header">
                <span>🏆 Head-to-Head Metric Benchmark</span>
              </div>

              <table className="data-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>MARL</th>
                    <th>Baseline</th>
                    <th>Winner</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Casualties Served</td>
                    <td>{result.marl_metrics.casualties_served}</td>
                    <td>{result.baseline_metrics.casualties_served}</td>
                    <td>
                      <span style={{ fontWeight: 700, color: result.marl_metrics.casualties_served >= result.baseline_metrics.casualties_served ? '#10b981' : '#f59e0b' }}>
                        {result.marl_metrics.casualties_served >= result.baseline_metrics.casualties_served ? '🏆 MARL' : 'Baseline'}
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td>Task Completion Rate</td>
                    <td>{(result.marl_metrics.task_completion_rate * 100).toFixed(1)}%</td>
                    <td>{(result.baseline_metrics.task_completion_rate * 100).toFixed(1)}%</td>
                    <td>
                      <span style={{ fontWeight: 700, color: result.marl_metrics.task_completion_rate >= result.baseline_metrics.task_completion_rate ? '#10b981' : '#f59e0b' }}>
                        {result.marl_metrics.task_completion_rate >= result.baseline_metrics.task_completion_rate ? '🏆 MARL' : 'Baseline'}
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td>Coordination Efficiency</td>
                    <td>{(result.marl_metrics.coordination_efficiency * 100).toFixed(1)}%</td>
                    <td>{(result.baseline_metrics.coordination_efficiency * 100).toFixed(1)}%</td>
                    <td>
                      <span style={{ fontWeight: 700, color: result.marl_metrics.coordination_efficiency >= result.baseline_metrics.coordination_efficiency ? '#10b981' : '#f59e0b' }}>
                        {result.marl_metrics.coordination_efficiency >= result.baseline_metrics.coordination_efficiency ? '🏆 MARL' : 'Baseline'}
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td>Avg Response Time</td>
                    <td>{result.marl_metrics.avg_response_time} steps</td>
                    <td>{result.baseline_metrics.avg_response_time} steps</td>
                    <td>
                      <span style={{ fontWeight: 700, color: result.marl_metrics.avg_response_time <= result.baseline_metrics.avg_response_time ? '#10b981' : '#f59e0b' }}>
                        {result.marl_metrics.avg_response_time <= result.baseline_metrics.avg_response_time ? '🏆 MARL' : 'Baseline'}
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="card">
              <div className="card-header">
                <span>📊 Relative Percentage Performance</span>
              </div>

              <div style={{ height: 250, width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="metric" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" unit="%" />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} />
                    <Legend />
                    <Bar dataKey="MARL" fill="#3b82f6" />
                    <Bar dataKey="Baseline" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
