/**
 * Minimal ECharts host. Modular imports keep the bundle to what we draw (lines, scatter; grid, tooltip, mark
 * lines/areas); the canvas renderer stays smooth while a replay updates every frame. Charts sharing a `group`
 * get linked crosshairs and tooltips.
 */
import { LineChart, ScatterChart } from "echarts/charts";
import { GridComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsCoreOption } from "echarts/core";
import { useEffect, useRef } from "react";

echarts.use([LineChart, ScatterChart, GridComponent, TooltipComponent, MarkLineComponent, MarkAreaComponent, CanvasRenderer]);

export type ChartOption = EChartsCoreOption;

interface Props {
  option: ChartOption;
  /** Text equivalent for screen readers (the chart is an image to them). */
  label: string;
  group?: string;
  className?: string;
  onSeriesClick?: (seriesId: string) => void;
}

export function EChart({ option, label, group, className, onSeriesClick }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const clickRef = useRef(onSeriesClick);
  clickRef.current = onSeriesClick;

  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current, undefined, { renderer: "canvas" });
    chart.current = c;
    if (group) {
      c.group = group;
      echarts.connect(group);
    }
    c.on("click", (p) => {
      const id = (p as { seriesId?: string }).seriesId;
      if (id) clickRef.current?.(id);
    });
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
    };
  }, [group]);

  useEffect(() => {
    // Series are replaced (not merged) so a deselected policy disappears; everything else merges.
    chart.current?.setOption(option, { replaceMerge: ["series"], lazyUpdate: true });
  }, [option]);

  return <div ref={el} role="img" aria-label={label} className={className} />;
}
