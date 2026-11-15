import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
  fetchRuns,
  fetchRunEpisodes,
  fetchTrainingCurve,
  fetchCheckpoints,
  fetchResultsSummary
} from '../api/client';
import { TrainingResults } from '../components/TrainingResults';

vi.mock('../api/client', () => ({
  fetchRuns: vi.fn(),
  fetchRunEpisodes: vi.fn(),
  fetchTrainingCurve: vi.fn(),
  fetchCheckpoints: vi.fn(),
  fetchResultsSummary: vi.fn()
}));

vi.mock('recharts', () => ({
  ResponsiveContainer: () => null,
  LineChart: () => null,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null
}));

describe('TrainingResults', () => {
  beforeEach(() => {
    vi.mocked(fetchRuns).mockResolvedValue([
      {
        run_id: 'latest-simulation',
        run_type: 'api_simulation',
        policy: 'marl',
        timestamp: '2026-11-15T12:00:00',
        notes: ''
      },
      {
        run_id: 'training-run',
        run_type: 'marl_training',
        policy: 'marl',
        timestamp: '2026-11-14T12:00:00',
        notes: 'Recorded training'
      }
    ]);
    vi.mocked(fetchRunEpisodes).mockResolvedValue([
      {
        episode_id: 1,
        run_id: 'training-run',
        episode_num: 1,
        total_reward: 12.5,
        avg_response_time: 4,
        resource_utilization_rate: 0.75,
        casualties_served: 3,
        casualties_lost: 1,
        casualties_total: 4,
        task_completion_rate: 0.75,
        coordination_efficiency: 0.8,
        medical_reward: 4,
        rescue_reward: 4,
        logistics_reward: 4.5
      }
    ]);
    vi.mocked(fetchTrainingCurve).mockResolvedValue([]);
    vi.mocked(fetchCheckpoints).mockResolvedValue([
      {
        filename: 'mappo_best.pt',
        path: 'checkpoints/mappo_best.pt',
        size_bytes: 1024,
        modified_time: '2026-11-15T12:00:00'
      }
    ]);
    vi.mocked(fetchResultsSummary).mockResolvedValue({
      summary_markdown: '',
      raw_data: [{ policy: 'marl', seed: 42, task_completion_rate: 0.75 }]
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('selects a training run and displays backend episode, checkpoint, and benchmark data', async () => {
    render(<TrainingResults />);

    expect(await screen.findByText(/Episode Results — training-run/)).toBeDefined();
    expect((screen.getByLabelText('Select run') as HTMLSelectElement).value).toBe('training-run');
    expect(screen.getByText('mappo_best.pt')).toBeDefined();
    expect(screen.getAllByText('12.5')).toHaveLength(2);
    expect(screen.getAllByText(/75\.0/).length).toBeGreaterThan(0);
    expect(screen.getByText('marl')).toBeDefined();
  });
});
