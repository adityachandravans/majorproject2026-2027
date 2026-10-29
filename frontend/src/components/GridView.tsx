import React from 'react';
import { SimulationFrame, AgentState, ZoneState, DepotState } from '../types';

interface GridViewProps {
  frame: SimulationFrame | null;
  title?: string;
  cellSize?: number;
  onCellClick?: (x: number, y: number) => void;
}

export const GridView: React.FC<GridViewProps> = ({ frame, title = 'Disaster Response Grid Map', cellSize = 60, onCellClick }) => {
  if (!frame) {
    return (
      <div className="svg-grid-container" style={{ minHeight: 300 }}>
        <p style={{ color: '#94a3b8' }}>No active simulation frame loaded.</p>
      </div>
    );
  }

  const [gridW, gridH] = frame.grid_size;
  const svgWidth = gridW * cellSize;
  const svgHeight = gridH * cellSize;

  const roleColors: Record<string, string> = {
    medical: '#ef4444',
    rescue: '#f59e0b',
    logistics: '#10b981'
  };

  const severityColors: Record<number, string> = {
    1: '#eab308',
    2: '#f97316',
    3: '#ef4444'
  };

  return (
    <div className="svg-grid-container" style={{ flexDirection: 'column' }}>
      <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', color: '#94a3b8', fontSize: '0.9rem' }}>
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>{title}</span>
        <span>Step {frame.step_count} / {frame.max_steps} ({frame.policy.toUpperCase()})</span>
      </div>

      <svg width={svgWidth} height={svgHeight} style={{ background: '#0b1329', borderRadius: 6, border: '1px solid #1e293b' }}>
        {/* 1. Grid Background & Lines */}
        {Array.from({ length: gridH }).map((_, y) =>
          Array.from({ length: gridW }).map((_, x) => (
            <rect
              key={`cell-${x}-${y}`}
              x={x * cellSize}
              y={y * cellSize}
              width={cellSize}
              height={cellSize}
              fill={(x + y) % 2 === 0 ? '#0d1936' : '#091228'}
              stroke="#1e293b"
              strokeWidth={1}
              style={{ cursor: onCellClick ? 'pointer' : 'default' }}
              onClick={() => onCellClick && onCellClick(x, y)}
            />
          ))
        )}

        {/* 2. Disaster Zones */}
        {frame.zones.map((z: ZoneState) => {
          const cx = z.pos[0] * cellSize + cellSize / 2;
          const cy = z.pos[1] * cellSize + cellSize / 2;
          const zoneSize = Math.max(1, z.size || 1);
          const coreRadius = cellSize * 0.38;
          const color = severityColors[z.severity] || '#ef4444';

          return (
            <g key={`zone-${z.id}`}>
              {/* Outer Area Footprint for Zones with size > 1 */}
              {zoneSize > 1 && (
                <rect
                  x={(z.pos[0] - (zoneSize - 1)) * cellSize + 2}
                  y={(z.pos[1] - (zoneSize - 1)) * cellSize + 2}
                  width={(zoneSize * 2 - 1) * cellSize - 4}
                  height={(zoneSize * 2 - 1) * cellSize - 4}
                  rx={8}
                  fill={color}
                  fillOpacity={0.12}
                  stroke={color}
                  strokeWidth={1.5}
                  strokeDasharray="4,4"
                  pointerEvents="none"
                />
              )}

              {/* Zone Center Circle */}
              <circle
                cx={cx}
                cy={cy}
                r={coreRadius}
                fill={color}
                fillOpacity={0.28}
                stroke={color}
                strokeWidth={z.accessibility ? 2 : 3}
                strokeDasharray={z.accessibility ? 'none' : '4,2'}
              />

              {/* Zone Label with Severity & Size */}
              <text x={cx} y={cy - 10} textAnchor="middle" fill="#ffffff" fontSize={11} fontWeight="bold">
                Z{z.id} (S{z.severity}{zoneSize > 1 ? ` · Sz:${zoneSize}` : ''})
              </text>

              {/* Casualty / Trapped Badges */}
              <text x={cx} y={cy + 4} textAnchor="middle" fill="#f8fafc" fontSize={10}>
                C:{z.casualty_count} | T:{z.trapped_count}
              </text>

              {/* Demand Info */}
              <text x={cx} y={cy + 18} textAnchor="middle" fill="#94a3b8" fontSize={9}>
                Dem:{z.supply_demand}
              </text>
            </g>
          );
        })}

        {/* 3. Depots */}
        {frame.depots.map((d: DepotState) => {
          const dx = d.pos[0] * cellSize + 6;
          const dy = d.pos[1] * cellSize + 6;
          const size = cellSize - 12;

          return (
            <g key={`depot-${d.id}`}>
              <rect
                x={dx}
                y={dy}
                width={size}
                height={size}
                rx={6}
                fill="#1e3a8a"
                fillOpacity={0.8}
                stroke="#3b82f6"
                strokeWidth={2}
              />
              <text x={dx + size / 2} y={dy + 16} textAnchor="middle" fill="#ffffff" fontSize={11} fontWeight="bold">
                Depot {d.id}
              </text>

              {/* Stock Bars */}
              <text x={dx + size / 2} y={dy + 30} textAnchor="middle" fill="#93c5fd" fontSize={9}>
                Amb:{d.ambulances} | Res:{d.rescue_teams}
              </text>
              <text x={dx + size / 2} y={dy + 42} textAnchor="middle" fill="#93c5fd" fontSize={9}>
                Sup:{d.supply_units}
              </text>
            </g>
          );
        })}

        {/* 4. Target Connection Lines */}
        {Object.entries(frame.agents).map(([aId, aData]: [string, AgentState]) => {
          if (aData.target_zone >= 0 && aData.target_zone < frame.zones.length) {
            const targetZ = frame.zones[aData.target_zone];
            const ax = aData.pos[0] * cellSize + cellSize / 2;
            const ay = aData.pos[1] * cellSize + cellSize / 2;
            const zx = targetZ.pos[0] * cellSize + cellSize / 2;
            const zy = targetZ.pos[1] * cellSize + cellSize / 2;
            const color = roleColors[aData.role] || '#ffffff';

            return (
              <line
                key={`line-${aId}`}
                x1={ax}
                y1={ay}
                x2={zx}
                y2={zy}
                stroke={color}
                strokeWidth={2}
                strokeDasharray="4 3"
                opacity={0.6}
              />
            );
          }
          return null;
        })}

        {/* 5. Agents */}
        {Object.entries(frame.agents).map(([aId, aData]: [string, AgentState], idx) => {
          const offsetShift = (idx % 3) * 6 - 6;
          const ax = aData.pos[0] * cellSize + cellSize / 2 + offsetShift;
          const ay = aData.pos[1] * cellSize + cellSize / 2 + offsetShift;
          const color = roleColors[aData.role] || '#a855f7';
          const roleChar = aData.role[0].toUpperCase();

          return (
            <g key={`agent-${aId}`} style={{ transition: 'all 0.3s ease' }}>
              <circle
                cx={ax}
                cy={ay}
                r={14}
                fill={color}
                stroke="#ffffff"
                strokeWidth={2}
              />
              <text
                x={ax}
                y={ay + 4}
                textAnchor="middle"
                fill="#ffffff"
                fontSize={11}
                fontWeight="bold"
              >
                {roleChar}
              </text>

              {/* Status / Payload Badge */}
              {aData.has_ambulance && <text x={ax + 10} y={ay - 10} fontSize={10}>🚑</text>}
              {aData.has_rescue_team && <text x={ax + 10} y={ay - 10} fontSize={10}>🛠️</text>}
              {aData.supply_payload > 0 && <text x={ax + 10} y={ay - 10} fontSize={10}>📦</text>}
            </g>
          );
        })}
      </svg>

      {/* Map Legend */}
      <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.75rem', fontSize: '0.8rem', color: '#94a3b8' }}>
        <span>🔴 <strong style={{ color: '#ef4444' }}>Medical (M)</strong></span>
        <span>🟡 <strong style={{ color: '#f59e0b' }}>Rescue (R)</strong></span>
        <span>🟢 <strong style={{ color: '#10b981' }}>Logistics (L)</strong></span>
        <span>🟦 <strong>Depot</strong></span>
        <span>⚠️ <strong>Dashed = Inaccessible Zone</strong></span>
      </div>
    </div>
  );
};
