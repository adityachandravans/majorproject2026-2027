import React, { useState, useEffect } from 'react';
import { fetchRuns, fetchRunEpisodes, fetchTrainingCurve, fetchCheckpoints, fetchResultsSummary } from '../api/client';
import { CheckpointInfo, RunRecord } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export const TrainingResults: React.FC = () => {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>('');
  const [curveData, setCurveData] = useState<any[]>([]);
  const [episodesData, setEpisodesData] = useState<any[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [summaryData, setSummaryData] = useState<{ summary_markdown: string; raw_data: any[] }>({
    summary_markdown: '',
    raw_data: []
  });

  useEffect(() => {
    fetchRuns().then((r) => {
      setRuns(r);
      if (r.length > 0) {
        setSelectedRunId(r[0].run_id);
      }
    }).catch(console.error);

    fetchCheckpoints().then(setCheckpoints).catch(console.error);
    fetchResultsSummary().then(setSummaryData).catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    fetchTrainingCurve(selectedRunId).then(setCurveData).catch(console.error);
    fetchRunEpisodes(selectedRunId).then(setEpisodesData).catch(console.error);
  }, [selectedRunId]);

  return (
    <div>
      <div className="grid-layout-2col">
        {/* 1. Training Learning Curve */}
        <div className="card">
          <div className="card-header">
            <span>📈 MAPPO Training Curve (MongoDB Logged)</span>
            {runs.length > 0 && (
              <select
                className="form-select"
                value={selectedRunId}
                onChange={(e) => setSelectedRunId(e.target.value)}
                style={{ width: 220 }}
              >
                {runs.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id} ({r.run_type})
                  </option>
                ))}
              </select>
            )}
          </div>

          <div style={{ height: 280, width: '100%' }}>
            {curveData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curveData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="update_num" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} />
                  <Legend />
                  <Line type="monotone" dataKey="mean_reward" name="Mean Reward" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="actor_loss" name="Actor Loss" stroke="#ef4444" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey="critic_loss" name="Critic Loss" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8' }}>
                No training curve data found for selected run.
              </div>
            )}
          </div>
        </div>

        {/* 2. Checkpoint Selection & Artifacts */}
        <div className="card">
          <div className="card-header">
            <span>💾 Saved Model Checkpoints</span>
          </div>

          <table className="data-table">
            <thead>
              <tr>
                <th>Checkpoint File</th>
                <th>Size</th>
                <th>Modified Date</th>
              </tr>
            </thead>
            <tbody>
              {checkpoints.length > 0 ? (
                checkpoints.map((c) => (
                  <tr key={c.filename}>
                    <td style={{ fontWeight: 600, color: '#38bdf8' }}>{c.filename}</td>
                    <td>{(c.size_bytes / 1024).toFixed(1)} KB</td>
                    <td>{new Date(c.modified_time).toLocaleString()}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={3} style={{ color: '#94a3b8' }}>No saved checkpoints found in checkpoints/</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. Evaluation Benchmark Results Table */}
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div className="card-header">
          <span>📊 Final Benchmark Evaluation Summary</span>
        </div>

        {summaryData.summary_markdown ? (
          <div style={{ padding: '1rem', background: '#0f172a', borderRadius: 6, border: '1px solid #334155', fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
            {summaryData.summary_markdown}
          </div>
        ) : (
          <p style={{ color: '#94a3b8' }}>Run evaluate.py to generate summary report.</p>
        )}
      </div>
    </div>
  );
};
