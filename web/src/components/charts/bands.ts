/**
 * A confidence band on a numeric x axis: one polygon through (x, hi) and back through (x, lo). Stacking two line
 * series only works on a category axis, so bands are drawn as a custom series; its data still sets the axis extents.
 */
export function bandSeries(id: string, points: { x: number; lo: number; hi: number }[], color: string, opacity = 0.14) {
  const pts = points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.lo) && Number.isFinite(p.hi));
  return {
    id, type: "custom", silent: true, z: 1, tooltip: { show: false },
    data: pts.map((p) => [p.x, p.lo, p.hi]),
    encode: { x: 0, y: [1, 2] },
    renderItem: (params: { dataIndex: number }, api: { coord: (v: [number, number]) => [number, number] }) => {
      if (params.dataIndex !== 0 || pts.length < 2) return null;
      const upper = pts.map((p) => api.coord([p.x, p.hi]));
      const lower = pts.map((p) => api.coord([p.x, p.lo])).reverse();
      return { type: "polygon", shape: { points: [...upper, ...lower] }, style: { fill: color, opacity }, silent: true };
    },
  }; // fmt: skip
}
