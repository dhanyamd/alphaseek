"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  LineStyle,
  AreaSeries,
} from "lightweight-charts";

export default function TVChart({
  data,
  height = 80,
  color = "#fafafa",
}: {
  data: number[];
  height?: number;
  color?: string;
}) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!container.current || !data || data.length < 2) return;
    const chart = createChart(container.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#5a5a5f",
        fontSize: 9,
      },
      grid: {
        vertLines: { color: "#1c1c1e", style: LineStyle.Solid },
        horzLines: { color: "#1c1c1e", style: LineStyle.Solid },
      },
      rightPriceScale: { visible: false },
      timeScale: { visible: false },
      handleScroll: false,
      handleScale: false,
    });

    const series = chart.addSeries(AreaSeries, {
      lineColor: color,
      topColor: color + "18",
      bottomColor: "transparent",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    series.setData(
      data.map((v, i) => ({ time: i as any, value: v }))
    );

    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [data, height, color]);

  if (!data || data.length < 2)
    return <div className="text-[11px] text-faint py-4 text-center mono">no data</div>;

  return <div ref={container} />;
}
