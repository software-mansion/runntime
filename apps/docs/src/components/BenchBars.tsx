/** Horizontal bar chart for a benchmark table: one bar per row, scaled to
 *  the longest. Renders at build time, no client script. ruNNtime rows are
 *  thick and bright, other engines thin and muted. */

export interface BenchBar {
  label: string;
  value: number;
  /** Shown after the value, e.g. `×` or ` ms`. */
  unit?: string;
  /** A ruNNtime row: drawn thick and bright. */
  ours?: boolean;
}

export function BenchBars({ rows, digits = 0 }: { rows: BenchBar[]; digits?: number }) {
  const max = Math.max(...rows.map((r) => r.value));
  return (
    <ul className="bench-bars">
      {rows.map((r) => (
        <li key={r.label} className={r.ours ? 'bench-bar-row bench-bar-row-ours' : 'bench-bar-row'}>
          <span className="bench-bar-label">{r.label}</span>
          <span className="bench-bar-track">
            <span className="bench-bar" style={{ width: `${(100 * r.value) / max}%` }} />
          </span>
          <span className="bench-bar-value">
            {r.value.toFixed(digits)}
            {r.unit ?? ''}
          </span>
        </li>
      ))}
    </ul>
  );
}
