import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { comparePolicies } from '../api/client';
import { ComparePolicies } from '../components/ComparePolicies';

vi.mock('../api/client', () => ({
  comparePolicies: vi.fn()
}));

describe('ComparePolicies', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it('shows backend comparison failures in the page', async () => {
    vi.mocked(comparePolicies).mockRejectedValue(
      new Error('Compare policies failed (500): Internal Server Error')
    );
    render(<ComparePolicies />);

    fireEvent.click(screen.getByRole('button', { name: /Run MARL vs Baseline Comparison/ }));

    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toContain('Compare policies failed (500): Internal Server Error');
  });
});
