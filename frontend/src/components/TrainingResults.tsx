import React, { useEffect, useState } from 'react';
import {
  fetchRuns,
  fetchRunEpisodes,
  fetchTrainingCurve,
  fetchCheckpoints,
  fetchResultsSummary
} from '../api/client';
import { CheckpointInfo, EpisodeRecord, EvaluationResult, ResultsSummary, RunRecord } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const initialSummary: ResultsSummary = { summary_markdown: '', raw_data: [] };

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatEvaluationValue(metric: string, value: EvaluationResult[string]): string {
  if (typeof value !== 'number') return value == null ? '—' : value;
  if (metric.endsWith('_rate')) return `${(value * 100).toFixed(1)}%`;
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

export const TrainingResults: React.FC = () => {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [curveData, setCurveData] = useState<any[]>([]);
  const [episodesData, setEpisodesData] = useState<EpisodeRecord[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [summaryData, setSummaryData] = useState<ResultsSummary>(initialSummary);
  const [loading, setLoading] = useState(true);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [detailsError, setDetailsError] = useState('');

  useEffect(() => {
    let active = true;

    const loadData = async () => {
      const [runsResult, checkpointsResult, summaryResult] = await Promise.allSettled([
        fetchRuns(),
        fetchCheckpoints(),
        fetchResultsSummary()
      ]);
      if (!active) return;

      const errors: string[] = [];
      if (runsResult.status === 'fulfilled') {
        const availableRuns = runsResult.value;
        setRuns(availableRuns);
        const preferredRun =
          availableRuns.find((run) => run.run_type.toLowerCase().includes('training')) ??
          availableRuns.find((run) => run.run_type !== 'api_simulation') ??
          availableRuns[0];
        setSelectedRunId(preferredRun?.run_id ?? '');
      } else {
        errors.push(`Runs: ${getErrorMessage(runsResult.reason)}`);
      }

      if (checkpointsResult.status === 'fulfilled') {
        setCheckpoints(checkpointsResult.value);
      } else {
        errors.push(`Checkpoints: ${getErrorMessage(checkpointsResult.reason)}`);
      }

      if (summaryResult.status === 'fulfilled') {
        setSummaryData(summaryResult.value);
      } else {
        errors.push(`Evaluation summary: ${getErrorMessage(summaryResult.reason)}`);
      }

      setLoadError(errors.join(' '));
      setLoading(false);
    };

    void loadData();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;

    let active = true;
    setDetailsLoading(true);
    setDetailsError('');
    setCurveData([]);
    setEpisodesData([]);

    const loadRunDetails = async () => {
      const [curveResult, episodesResult] = await Promise.allSettled([
        fetchTrainingCurve(selectedRunId),
        fetchRunEpisodes(selectedRunId)
      ]);
      if (!active) return;

      const errors: string[] = [];
      if (curveResult.status === 'fulfilled') {
        setCurveData(curveResult.value);
      } else {
        errors.push(`Training curve: ${getErrorMessage(curveResult.reason)}`);
      }

      if (episodesResult.status === 'fulfilled') {
        setEpisodesData(episodesResult.value);
      } else {
        errors.push(`Episode results: ${getErrorMessage(episodesResult.reason)}`);
      }

      setDetailsError(errors.join(' '));
      setDetailsLoading(false);
    };

    void loadRunDetails();
    return () => {
      active = false;
    };
  }, [selectedRunId]);

  const selectedRun = runs.find((run) => run.run_id === selectedRunId);
  const progressData = curveData.length > 0
    ? curveData
    : episodesData.map((episode) => ({
        update_num: episode.episode_num,
        mean_reward: episode.total_reward
      }));
  const averageReward = episodesData.length
    ? episodesData.reduce((total, episode) => total + episode.total_reward, 0) / episodesData.length
    : 0;
  const averageCompletion = episodesData.length
    ? episodesData.reduce((total, episode) => total + episode.task_completion_rate, 0) / episodesData.length
    : 0;
  const casualtiesServed = episodesData.reduce((total, episode) => total + episode.casualties_served, 0);
  const evaluationColumns = summaryData.raw_data.length
    ? Object.keys(summaryData.raw_data[0])
    : [];

  return (
    <div>
      <h2 style={{ marginBottom: '0.5rem' }}>Training & Evaluation Results</h2>
      <p style={{ color: '#94a3b8', marginBottom: '1.5rem' }}>
        Run history, episode outcomes, saved checkpoints, and benchmark results loaded from the backend.
      </p>

      {loadError && (
        <div role="alert" style={{ background: '#7f1d1d', color: '#fecaca', padding: '0.75rem', borderRadius: 6, marginBottom: '1rem' }}>
          {loadError}
        </div>
      )}
      {detailsError && (
        <div role="alert" style={{ background: '#7f1d1d', color: '#fecaca', padding: '0.75rem', borderRadius: 6, marginBottom: '1rem' }}>
          {detailsError}
        </div>
      )}
      {loading && <p role="status" style={{ color: '#94a3b8', marginBottom: '1rem' }}>Loading results from backend…</p>}
      {!loading && runs.length === 0 && (
        <div className="card">
          No recorded runs are available yet. Complete a simulation or run training to create results.
        </div>
      )}

      <div className="grid-layout-2col">
        <div className="card">
          <div className="card-header">
            <span>📈 Training Progress</span>
            {runs.length > 0 && (
              <select
                aria-label="Select run"
                className="form-select"
                value={selectedRunId}
                onChange={(event) => setSelectedRunId(event.target.value)}
                style={{ width: 240 }}
              >
                {runs.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_id} ({run.run_type})
                  </option>
                ))}
              </select>
            )}
          </div>
          {selectedRun && (
            <p style={{ color: '#94a3b8', marginBottom: '0.75rem' }}>
              {selectedRun.policy} · {new Date(selectedRun.timestamp).toLocaleString()}
              {selectedRun.notes ? ` · ${selectedRun.notes}` : ''}
            </p>
          )}

          <div style={{ height: 280, width: '100%' }}>
            {detailsLoading ? (
              <div role="status" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8' }}>
                Loading run metrics…
              </div>
            ) : progressData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={progressData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="update_num" stroke="#94a3b8" label={{ value: curveData.length ? 'Training update' : 'Episode', position: 'insideBottom', offset: -5 }} />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="mean_reward"
                    name={curveData.length ? 'Mean update reward' : 'Episode reward'}
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                  {curveData.length > 0 && (
                    <>
                      <Line type="monotone" dataKey="actor_loss" name="Actor loss" stroke="#ef4444" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="critic_loss" name="Critic loss" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                    </>
                  )}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8', textAlign: 'center' }}>
                No update-level curve or episode metrics have been recorded for this run.
              </div>
            )}
          </div>
          {!detailsLoading && curveData.length === 0 && episodesData.length > 0 && (
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '0.5rem' }}>
              Update-level losses were not recorded for this run; the chart shows recorded episode rewards.
            </p>
          )}
        </div>

        <div className="card">
          <div className="card-header"><span>💾 Saved Model Checkpoints</span></div>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr><th>Checkpoint</th><th>Size</th><th>Modified</th></tr>
              </thead>
              <tbody>
                {checkpoints.length > 0 ? checkpoints.map((checkpoint) => (
                  <tr key={checkpoint.filename}>
                    <td style={{ fontWeight: 600, color: '#38bdf8' }}>{checkpoint.filename}</td>
                    <td>{(checkpoint.size_bytes / 1024).toFixed(1)} KB</td>
                    <td>{new Date(checkpoint.modified_time).toLocaleString()}</td>
                  </tr>
                )) : (
                  <tr><td colSpan={3} style={{ color: '#94a3b8' }}>No saved checkpoints found.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {episodesData.length > 0 && (
        <>
          <div className="grid-layout-3col">
            <div className="metric-card">
              <div className="metric-label">Recorded episodes</div>
              <div className="metric-value">{episodesData.length}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Average episode reward</div>
              <div className="metric-value">{averageReward.toFixed(1)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Average task completion</div>
              <div className="metric-value">{(averageCompletion * 100).toFixed(1)}%</div>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span>📋 Episode Results — {selectedRun?.run_id}</span>
              <span style={{ color: '#94a3b8', fontSize: '0.9rem' }}>{casualtiesServed} casualties served across episodes</span>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Episode</th><th>Total reward</th><th>Response time</th>
                    <th>Task completion</th><th>Coordination</th><th>Casualties served</th>
                    <th>Casualties lost</th><th>Casualties total</th>
                  </tr>
                </thead>
                <tbody>
                  {episodesData.map((episode) => (
                    <tr key={episode.episode_id}>
                      <td>{episode.episode_num}</td>
                      <td>{episode.total_reward.toFixed(1)}</td>
                      <td>{episode.avg_response_time.toFixed(1)}</td>
                      <td>{(episode.task_completion_rate * 100).toFixed(1)}%</td>
                      <td>{(episode.coordination_efficiency * 100).toFixed(1)}%</td>
                      <td>{episode.casualties_served}</td>
                      <td>{episode.casualties_lost}</td>
                      <td>{episode.casualties_total}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
      {!detailsLoading && selectedRunId && episodesData.length === 0 && (
        <div className="card" role="status">
          No episode results have been recorded for this run yet.
        </div>
      )}

      <div className="card">
        <div className="card-header"><span>📊 Benchmark Evaluation Results</span></div>
        {evaluationColumns.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>{evaluationColumns.map((column) => <th key={column}>{column.replace(/_/g, ' ')}</th>)}</tr>
              </thead>
              <tbody>
                {summaryData.raw_data.map((result, index) => (
                  <tr key={`${String(result.policy ?? 'result')}-${String(result.seed ?? index)}-${index}`}>
                    {evaluationColumns.map((column) => (
                      <td key={column}>{formatEvaluationValue(column, result[column])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ color: '#94a3b8' }}>
            No benchmark CSV data is available on the backend. Run metrics are shown above when present.
          </p>
        )}
        {summaryData.summary_markdown && (
          <pre style={{ marginTop: '1rem', padding: '1rem', background: '#0f172a', borderRadius: 6, border: '1px solid #334155', whiteSpace: 'pre-wrap', overflowX: 'auto' }}>
            {summaryData.summary_markdown}
          </pre>
        )}
      </div>
    </div>
  );
};
