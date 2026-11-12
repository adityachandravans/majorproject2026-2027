import { afterEach, describe, expect, it, vi } from 'vitest';
import { createSimulation } from '../api/client';

describe('API client errors', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('includes the HTTP status and response detail when simulation creation fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => 'Internal Server Error'
    }));

    await expect(createSimulation({ policy: 'baseline' }))
      .rejects.toThrow('Create simulation failed (500): Internal Server Error');
  });
});
