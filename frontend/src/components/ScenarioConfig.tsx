import React, { useState, useEffect } from 'react';
import { fetchDefaultConfig, createSimulation } from '../api/client';

export interface CustomZoneConfig {
  id: number;
  pos: [number, number];
  size: number;
  severity: number;
  casualty_count: number;
  trapped_count: number;
  accessibility: boolean;
  supply_demand?: number;
}

interface ScenarioConfigProps {
  onSimulationCreated: (simId: string, cfg?: any) => void;
}

export const ScenarioConfig: React.FC<ScenarioConfigProps> = ({ onSimulationCreated }) => {
  const [config, setConfig] = useState<any>(null);
  const [seed, setSeed] = useState<number>(42);
  const [policy, setPolicy] = useState<string>('marl');
  const [mode, setMode] = useState<'custom' | 'generator'>('custom');

  // Custom zones list
  const [customZones, setCustomZones] = useState<CustomZoneConfig[]>([
    { id: 0, pos: [2, 2], size: 1, severity: 2, casualty_count: 5, trapped_count: 3, accessibility: true },
    { id: 1, pos: [6, 3], size: 2, severity: 3, casualty_count: 8, trapped_count: 4, accessibility: false },
    { id: 2, pos: [4, 7], size: 3, severity: 1, casualty_count: 3, trapped_count: 1, accessibility: true }
  ]);
  const [selectedZoneId, setSelectedZoneId] = useState<number>(0);

  // Dynamic Generator options
  const [randomizeSizes, setRandomizeSizes] = useState<boolean>(false);
  const [minSize, setMinSize] = useState<number>(1);
  const [maxSize, setMaxSize] = useState<number>(3);
  const [dynamicSizeEvolution, setDynamicSizeEvolution] = useState<boolean>(true);
  const [sizeChangeProb, setSizeChangeProb] = useState<number>(0.15);

  useEffect(() => {
    fetchDefaultConfig().then((cfg) => {
      setConfig(cfg);
      if (cfg?.environment?.zone_size_range) {
        setMinSize(cfg.environment.zone_size_range[0] || 1);
        setMaxSize(cfg.environment.zone_size_range[1] || 3);
      }
      if (cfg?.environment?.dynamic_zone_sizes !== undefined) {
        setDynamicSizeEvolution(cfg.environment.dynamic_zone_sizes);
      }
    }).catch(console.error);
  }, []);

  const gridW = config?.environment?.grid_size?.[0] || 10;
  const gridH = config?.environment?.grid_size?.[1] || 10;

  // Add new zone
  const handleAddZone = () => {
    const nextId = customZones.length > 0 ? Math.max(...customZones.map(z => z.id)) + 1 : 0;
    // Find unoccupied or random coordinate
    const rx = Math.floor(Math.random() * gridW);
    const ry = Math.floor(Math.random() * gridH);
    const newZone: CustomZoneConfig = {
      id: nextId,
      pos: [rx, ry],
      size: 1,
      severity: 2,
      casualty_count: 4,
      trapped_count: 2,
      accessibility: true
    };
    setCustomZones([...customZones, newZone]);
    setSelectedZoneId(nextId);
  };

  // Remove zone
  const handleRemoveZone = (id: number) => {
    if (customZones.length <= 1) return;
    const filtered = customZones.filter(z => z.id !== id);
    setCustomZones(filtered);
    if (selectedZoneId === id) {
      setSelectedZoneId(filtered[0]?.id ?? 0);
    }
  };

  // Update specific zone field
  const handleUpdateZone = (id: number, field: keyof CustomZoneConfig, value: any) => {
    setCustomZones(customZones.map(z => {
      if (z.id === id) {
        return { ...z, [field]: value };
      }
      return z;
    }));
  };

  // Grid cell click to reposition or place selected zone
  const handleCellClick = (x: number, y: number) => {
    if (selectedZoneId !== null) {
      handleUpdateZone(selectedZoneId, 'pos', [x, y]);
    }
  };

  // Randomize all zone sizes
  const handleRandomizeSizes = () => {
    setCustomZones(customZones.map(z => ({
      ...z,
      size: Math.floor(Math.random() * 3) + 1
    })));
  };

  // Randomize all zone positions & casualties
  const handleRandomizeAll = () => {
    setCustomZones(customZones.map(z => ({
      ...z,
      pos: [Math.floor(Math.random() * gridW), Math.floor(Math.random() * gridH)],
      size: Math.floor(Math.random() * 3) + 1,
      severity: Math.floor(Math.random() * 3) + 1,
      casualty_count: Math.floor(Math.random() * 8) + 1,
      trapped_count: Math.floor(Math.random() * 5) + 1,
      accessibility: Math.random() > 0.25
    })));
  };

  // Preset scenarios
  const handlePreset = (type: string) => {
    if (!config) return;
    const newCfg = JSON.parse(JSON.stringify(config));

    if (type === 'earthquake') {
      newCfg.environment.grid_size = [12, 12];
      newCfg.environment.agent_counts = { medical: 3, rescue: 4, logistics: 2 };
      setCustomZones([
        { id: 0, pos: [3, 3], size: 3, severity: 3, casualty_count: 12, trapped_count: 8, accessibility: false },
        { id: 1, pos: [8, 4], size: 2, severity: 3, casualty_count: 9, trapped_count: 6, accessibility: true },
        { id: 2, pos: [4, 9], size: 2, severity: 2, casualty_count: 6, trapped_count: 3, accessibility: true },
        { id: 3, pos: [9, 9], size: 1, severity: 1, casualty_count: 3, trapped_count: 1, accessibility: true }
      ]);
      setSelectedZoneId(0);
    } else if (type === 'flood') {
      newCfg.environment.grid_size = [10, 10];
      newCfg.environment.agent_counts = { medical: 2, rescue: 3, logistics: 3 };
      setCustomZones([
        { id: 0, pos: [2, 5], size: 3, severity: 2, casualty_count: 6, trapped_count: 5, accessibility: false },
        { id: 1, pos: [7, 5], size: 2, severity: 2, casualty_count: 5, trapped_count: 4, accessibility: false },
        { id: 2, pos: [5, 2], size: 2, severity: 1, casualty_count: 4, trapped_count: 2, accessibility: true }
      ]);
      setSelectedZoneId(0);
    } else if (type === 'cyclone') {
      newCfg.environment.grid_size = [14, 14];
      newCfg.environment.agent_counts = { medical: 4, rescue: 3, logistics: 4 };
      setCustomZones([
        { id: 0, pos: [4, 4], size: 4, severity: 3, casualty_count: 15, trapped_count: 10, accessibility: false },
        { id: 1, pos: [9, 3], size: 2, severity: 2, casualty_count: 7, trapped_count: 4, accessibility: true },
        { id: 2, pos: [3, 10], size: 3, severity: 3, casualty_count: 10, trapped_count: 7, accessibility: false },
        { id: 3, pos: [10, 10], size: 2, severity: 1, casualty_count: 5, trapped_count: 2, accessibility: true },
        { id: 4, pos: [7, 7], size: 1, severity: 2, casualty_count: 4, trapped_count: 2, accessibility: true }
      ]);
      setSelectedZoneId(0);
    } else if (type === 'single_super_hazard') {
      newCfg.environment.grid_size = [10, 10];
      newCfg.environment.agent_counts = { medical: 3, rescue: 3, logistics: 3 };
      setCustomZones([
        { id: 0, pos: [5, 5], size: 4, severity: 3, casualty_count: 25, trapped_count: 15, accessibility: false }
      ]);
      setSelectedZoneId(0);
    }

    setConfig(newCfg);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!config) return;

    const payloadConfig = JSON.parse(JSON.stringify(config));
    payloadConfig.environment.dynamic_zone_sizes = dynamicSizeEvolution;
    payloadConfig.environment.size_change_prob = sizeChangeProb;
    payloadConfig.environment.zone_size_range = [minSize, maxSize];

    if (mode === 'custom') {
      payloadConfig.environment.custom_zones = customZones;
      payloadConfig.environment.num_zones = customZones.length;
      payloadConfig.environment.randomize_zone_sizes = false;
    } else {
      delete payloadConfig.environment.custom_zones;
      payloadConfig.environment.randomize_zone_sizes = randomizeSizes;
    }

    try {
      const res = await createSimulation({ config: payloadConfig, policy, seed });
      onSimulationCreated(res.sim_id, payloadConfig);
    } catch (err) {
      console.error('Failed to create simulation:', err);
    }
  };

  if (!config) return <div className="card">Loading configuration parameters...</div>;

  const env = config.environment;
  const comms = config.comms || { comm_enabled: true };
  const selectedZone = customZones.find(z => z.id === selectedZoneId) || customZones[0];

  const severityColors: Record<number, string> = {
    1: '#eab308',
    2: '#f97316',
    3: '#ef4444'
  };

  return (
    <div className="card" style={{ maxWidth: 1000, margin: '0 auto' }}>
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>⚙️ Dynamic Scenario & Zone Designer</span>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            type="button"
            className={`btn ${mode === 'custom' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ fontSize: '0.85rem', padding: '4px 12px' }}
            onClick={() => setMode('custom')}
          >
            🎨 Custom Zone Designer ({customZones.length} Zones)
          </button>
          <button
            type="button"
            className={`btn ${mode === 'generator' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ fontSize: '0.85rem', padding: '4px 12px' }}
            onClick={() => setMode('generator')}
          >
            🎲 Dynamic Generator
          </button>
        </div>
      </div>

      {/* Quick Presets */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.25rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#94a3b8' }}>Presets:</span>
        <button type="button" className="btn btn-secondary" style={{ fontSize: '0.8rem', padding: '4px 10px' }} onClick={() => handlePreset('earthquake')}>🌋 Earthquake (4 Zones, High Damage)</button>
        <button type="button" className="btn btn-secondary" style={{ fontSize: '0.8rem', padding: '4px 10px' }} onClick={() => handlePreset('flood')}>🌊 Flood (3 Large Zones)</button>
        <button type="button" className="btn btn-secondary" style={{ fontSize: '0.8rem', padding: '4px 10px' }} onClick={() => handlePreset('cyclone')}>🌀 Cyclone (5 Dynamic Zones)</button>
        <button type="button" className="btn btn-secondary" style={{ fontSize: '0.8rem', padding: '4px 10px' }} onClick={() => handlePreset('single_super_hazard')}>💥 Super Hazard (1 Massive Zone)</button>
      </div>

      <form onSubmit={handleSubmit}>
        {/* CUSTOM ZONE DESIGNER MODE */}
        {mode === 'custom' && (
          <div style={{ background: '#0b1329', padding: '1rem', borderRadius: 8, border: '1px solid #1e293b', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div>
                <h4 style={{ color: '#38bdf8', margin: 0 }}>Interactive Zone Builder</h4>
                <p style={{ margin: '4px 0 0', fontSize: '0.8rem', color: '#94a3b8' }}>
                  Create any number of zones with customizable size, position, casualties, and blockage. Click grid cells to reposition!
                </p>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button type="button" className="btn btn-secondary" style={{ fontSize: '0.8rem' }} onClick={handleRandomizeSizes}>
                  🎲 Randomize Sizes
                </button>
                <button type="button" className="btn btn-secondary" style={{ fontSize: '0.8rem' }} onClick={handleRandomizeAll}>
                  🎲 Randomize All
                </button>
                <button type="button" className="btn btn-primary" style={{ fontSize: '0.8rem' }} onClick={handleAddZone}>
                  ➕ Add Zone
                </button>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 340px) 1fr', gap: '1.25rem' }}>
              {/* Interactive Mini Map */}
              <div>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f8fafc', marginBottom: '0.5rem' }}>
                  Grid Preview (Click cell to place Zone {selectedZone?.id})
                </div>
                <div style={{ display: 'inline-block', background: '#070d1e', padding: 8, borderRadius: 6, border: '1px solid #1e293b' }}>
                  <svg
                    width={gridW * 26}
                    height={gridH * 26}
                    style={{ background: '#091228', display: 'block', borderRadius: 4 }}
                  >
                    {/* Grid Cells */}
                    {Array.from({ length: gridH }).map((_, y) =>
                      Array.from({ length: gridW }).map((_, x) => {
                        const isSelectedZonePos = selectedZone?.pos[0] === x && selectedZone?.pos[1] === y;
                        return (
                          <rect
                            key={`map-cell-${x}-${y}`}
                            x={x * 26}
                            y={y * 26}
                            width={26}
                            height={26}
                            fill={isSelectedZonePos ? '#1e3a8a' : (x + y) % 2 === 0 ? '#0d1936' : '#081226'}
                            stroke="#1e293b"
                            strokeWidth={1}
                            style={{ cursor: 'pointer' }}
                            onClick={() => handleCellClick(x, y)}
                          />
                        );
                      })
                    )}

                    {/* Render Zones */}
                    {customZones.map(z => {
                      const cx = z.pos[0] * 26 + 13;
                      const cy = z.pos[1] * 26 + 13;
                      const isSel = z.id === selectedZoneId;
                      const color = severityColors[z.severity] || '#ef4444';
                      const zSize = Math.max(1, z.size);

                      return (
                        <g key={`preview-zone-${z.id}`} onClick={() => setSelectedZoneId(z.id)} style={{ cursor: 'pointer' }}>
                          {/* Footprint if size > 1 */}
                          {zSize > 1 && (
                            <rect
                              x={(z.pos[0] - (zSize - 1)) * 26 + 2}
                              y={(z.pos[1] - (zSize - 1)) * 26 + 2}
                              width={(zSize * 2 - 1) * 26 - 4}
                              height={(zSize * 2 - 1) * 26 - 4}
                              rx={4}
                              fill={color}
                              fillOpacity={0.15}
                              stroke={color}
                              strokeWidth={1}
                              strokeDasharray="2,2"
                            />
                          )}

                          <circle
                            cx={cx}
                            cy={cy}
                            r={isSel ? 10 : 8}
                            fill={color}
                            fillOpacity={0.4}
                            stroke={isSel ? '#ffffff' : color}
                            strokeWidth={isSel ? 2.5 : 1.5}
                          />
                          <text x={cx} y={cy + 3.5} textAnchor="middle" fill="#ffffff" fontSize={9} fontWeight="bold">
                            Z{z.id}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                </div>
                <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#94a3b8' }}>
                  Total Zones: <strong style={{ color: '#38bdf8' }}>{customZones.length}</strong> | Total Casualties: <strong style={{ color: '#f87171' }}>{customZones.reduce((s, z) => s + z.casualty_count, 0)}</strong> | Trapped: <strong style={{ color: '#fbbf24' }}>{customZones.reduce((s, z) => s + z.trapped_count, 0)}</strong>
                </div>
              </div>

              {/* Zone List & Selected Zone Editor */}
              <div>
                {/* Zone Selector Pills */}
                <div style={{ display: 'flex', gap: '0.35rem', overflowX: 'auto', marginBottom: '0.75rem', paddingBottom: '4px' }}>
                  {customZones.map(z => (
                    <button
                      key={`pill-${z.id}`}
                      type="button"
                      onClick={() => setSelectedZoneId(z.id)}
                      className={`btn ${selectedZoneId === z.id ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ fontSize: '0.8rem', padding: '4px 10px', whiteSpace: 'nowrap' }}
                    >
                      Z{z.id} (Sz:{z.size}, C:{z.casualty_count})
                    </button>
                  ))}
                </div>

                {/* Edit Selected Zone */}
                {selectedZone && (
                  <div style={{ background: '#091228', padding: '0.75rem', borderRadius: 6, border: '1px solid #1e293b' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                      <span style={{ fontWeight: 600, color: '#f8fafc', fontSize: '0.9rem' }}>
                        Zone {selectedZone.id} Parameters
                      </span>
                      <button
                        type="button"
                        onClick={() => handleRemoveZone(selectedZone.id)}
                        disabled={customZones.length <= 1}
                        style={{ background: '#dc2626', color: '#fff', border: 'none', borderRadius: 4, padding: '3px 8px', fontSize: '0.75rem', cursor: customZones.length <= 1 ? 'not-allowed' : 'pointer' }}
                      >
                        🗑️ Delete Zone
                      </button>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem' }}>
                      <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">Position [X, Y]</label>
                        <div style={{ display: 'flex', gap: '0.4rem' }}>
                          <input
                            type="number"
                            min={0}
                            max={gridW - 1}
                            className="form-input"
                            value={selectedZone.pos[0]}
                            onChange={(e) => handleUpdateZone(selectedZone.id, 'pos', [parseInt(e.target.value) || 0, selectedZone.pos[1]])}
                          />
                          <input
                            type="number"
                            min={0}
                            max={gridH - 1}
                            className="form-input"
                            value={selectedZone.pos[1]}
                            onChange={(e) => handleUpdateZone(selectedZone.id, 'pos', [selectedZone.pos[0], parseInt(e.target.value) || 0])}
                          />
                        </div>
                      </div>

                      <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">
                          Zone Size (Radius: {selectedZone.size - 1} cells)
                        </label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <input
                            type="range"
                            min={1}
                            max={5}
                            step={1}
                            value={selectedZone.size}
                            onChange={(e) => handleUpdateZone(selectedZone.id, 'size', parseInt(e.target.value) || 1)}
                            style={{ flex: 1 }}
                          />
                          <span style={{ fontWeight: 700, color: '#38bdf8', minWidth: 45 }}>
                            Size {selectedZone.size}
                          </span>
                        </div>
                      </div>

                      <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">Casualties Count</label>
                        <input
                          type="number"
                          min={0}
                          max={50}
                          className="form-input"
                          value={selectedZone.casualty_count}
                          onChange={(e) => handleUpdateZone(selectedZone.id, 'casualty_count', parseInt(e.target.value) || 0)}
                        />
                      </div>

                      <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">Trapped People</label>
                        <input
                          type="number"
                          min={0}
                          max={50}
                          className="form-input"
                          value={selectedZone.trapped_count}
                          onChange={(e) => handleUpdateZone(selectedZone.id, 'trapped_count', parseInt(e.target.value) || 0)}
                        />
                      </div>

                      <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">Severity Level</label>
                        <select
                          className="form-select"
                          value={selectedZone.severity}
                          onChange={(e) => handleUpdateZone(selectedZone.id, 'severity', parseInt(e.target.value) || 1)}
                        >
                          <option value={1}>1 - Low Hazard</option>
                          <option value={2}>2 - Medium Hazard</option>
                          <option value={3}>3 - Critical Hazard</option>
                        </select>
                      </div>

                      <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">Accessibility / Rubble</label>
                        <select
                          className="form-select"
                          value={selectedZone.accessibility ? 'true' : 'false'}
                          onChange={(e) => handleUpdateZone(selectedZone.id, 'accessibility', e.target.value === 'true')}
                        >
                          <option value="true">Open (Accessible)</option>
                          <option value="false">Blocked / Inaccessible (+Delay)</option>
                        </select>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* DYNAMIC GENERATOR MODE */}
        {mode === 'generator' && (
          <div style={{ background: '#0b1329', padding: '1rem', borderRadius: 8, border: '1px solid #1e293b', marginBottom: '1.5rem' }}>
            <h4 style={{ color: '#38bdf8', marginBottom: '0.75rem' }}>Procedural Generator & Size Randomization</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
              <div className="form-group">
                <label className="form-label">Number of Disaster Zones</label>
                <input
                  type="number"
                  min={1}
                  max={15}
                  className="form-input"
                  value={env.num_zones}
                  onChange={(e) =>
                    setConfig({ ...config, environment: { ...env, num_zones: parseInt(e.target.value) || 4 } })
                  }
                />
              </div>

              <div className="form-group">
                <label className="form-label">Randomize Initial Zone Sizes?</label>
                <select
                  className="form-select"
                  value={randomizeSizes ? 'true' : 'false'}
                  onChange={(e) => setRandomizeSizes(e.target.value === 'true')}
                >
                  <option value="false">Uniform Size (Default: Size 1)</option>
                  <option value="true">Random Sizes per Zone</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Zone Size Range [Min - Max]</label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <input
                    type="number"
                    min={1}
                    max={5}
                    className="form-input"
                    value={minSize}
                    onChange={(e) => setMinSize(parseInt(e.target.value) || 1)}
                  />
                  <input
                    type="number"
                    min={1}
                    max={5}
                    className="form-input"
                    value={maxSize}
                    onChange={(e) => setMaxSize(parseInt(e.target.value) || 3)}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* DYNAMIC RUNTIME SIZE EVOLUTION */}
        <div style={{ background: '#0b1329', padding: '1rem', borderRadius: 8, border: '1px solid #1e293b', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h4 style={{ color: '#38bdf8', margin: 0 }}>🌊 Dynamic Zone Size Changes During Simulation</h4>
              <p style={{ margin: '4px 0 0', fontSize: '0.8rem', color: '#94a3b8' }}>
                Disaster zones expand or contract dynamically over time as the situation develops.
              </p>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontWeight: 600 }}>
              <input
                type="checkbox"
                checked={dynamicSizeEvolution}
                onChange={(e) => setDynamicSizeEvolution(e.target.checked)}
                style={{ width: 18, height: 18 }}
              />
              <span style={{ color: dynamicSizeEvolution ? '#38bdf8' : '#94a3b8' }}>
                {dynamicSizeEvolution ? 'Enabled' : 'Disabled'}
              </span>
            </label>
          </div>

          {dynamicSizeEvolution && (
            <div style={{ display: 'flex', gap: '1rem', marginTop: '0.75rem', alignItems: 'center' }}>
              <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Fluctuation Probability:</span>
              <input
                type="range"
                min="0.05"
                max="0.50"
                step="0.05"
                value={sizeChangeProb}
                onChange={(e) => setSizeChangeProb(parseFloat(e.target.value))}
                style={{ width: 160 }}
              />
              <span style={{ color: '#38bdf8', fontSize: '0.85rem', fontWeight: 600 }}>
                {Math.round(sizeChangeProb * 100)}% chance per step
              </span>
            </div>
          )}
        </div>

        {/* ENVIRONMENT SCALE & AGENTS */}
        <div className="grid-layout-2col">
          {/* Environment Scale */}
          <div>
            <h4 style={{ color: '#38bdf8', marginBottom: '0.75rem' }}>Grid & Time Specs</h4>
            <div className="form-group">
              <label className="form-label">Grid Width / Height</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <input
                  type="number"
                  min={5}
                  max={30}
                  className="form-input"
                  value={env.grid_size[0]}
                  onChange={(e) => {
                    const val = parseInt(e.target.value) || 10;
                    setConfig({ ...config, environment: { ...env, grid_size: [val, env.grid_size[1]] } });
                  }}
                />
                <input
                  type="number"
                  min={5}
                  max={30}
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
              <label className="form-label">Depots Count</label>
              <input
                type="number"
                min={1}
                max={4}
                className="form-input"
                value={env.num_depots}
                onChange={(e) =>
                  setConfig({ ...config, environment: { ...env, num_depots: parseInt(e.target.value) || 2 } })
                }
              />
            </div>
          </div>

          {/* Agents */}
          <div>
            <h4 style={{ color: '#38bdf8', marginBottom: '0.75rem' }}>Agents & Communication</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem' }}>
              <div className="form-group">
                <label className="form-label">Medical</label>
                <input
                  type="number"
                  min={1}
                  max={10}
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
                <label className="form-label">Rescue</label>
                <input
                  type="number"
                  min={1}
                  max={10}
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
                <label className="form-label">Logistics</label>
                <input
                  type="number"
                  min={1}
                  max={10}
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

        {/* SUBMIT BUTTON BAR */}
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

          <div style={{ flex: 1.2, display: 'flex', alignItems: 'flex-end' }}>
            <button type="submit" className="btn btn-primary" style={{ width: '100%', height: 40, marginTop: 24, fontSize: '0.95rem', fontWeight: 600 }}>
              🚀 Launch Dynamic Scenario ({mode === 'custom' ? `${customZones.length} Zones` : `${env.num_zones} Zones`})
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};
