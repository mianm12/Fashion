<template>
  <div class="heat-chart-wrap">
    <div v-if="points.length === 0" class="dense-state chart-empty">
      暂无最近 8 周热度和增长序列
    </div>
    <VChart v-else class="heat-chart" :option="option" autoresize />
  </div>
</template>

<script setup lang="ts">
import { LineChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import VChart from "vue-echarts";
import { computed } from "vue";

import type { HeatSeriesPoint } from "@/types/api";
import { formatNumber, formatSignedPercent } from "@/utils/formatters";

use([
  CanvasRenderer,
  GridComponent,
  LegendComponent,
  LineChart,
  MarkLineComponent,
  TooltipComponent,
]);

const props = defineProps<{
  points: HeatSeriesPoint[];
  sourceWeek?: number | null;
}>();

const option = computed(() => {
  const ordered = [...props.points].sort((left, right) => left.week_id - right.week_id);

  return {
    color: ["#3A5A6A", "#B42318", "#9A6700"],
    grid: {
      left: 42,
      right: 46,
      top: 36,
      bottom: 32,
    },
    legend: {
      top: 0,
      itemWidth: 16,
      itemHeight: 8,
      textStyle: { color: "#636b65", fontSize: 11 },
    },
    tooltip: {
      trigger: "axis",
      formatter: (params: unknown) => formatTooltip(params, ordered),
    },
    xAxis: {
      type: "category",
      boundaryGap: false,
      data: ordered.map((point) => `W${point.week_id}`),
      axisLabel: { color: "#636b65", fontSize: 11 },
      axisLine: { lineStyle: { color: "#d7dcd5" } },
      axisTick: { show: false },
    },
    yAxis: [
      {
        type: "value",
        name: "热度",
        nameTextStyle: { color: "#636b65", fontSize: 11 },
        axisLabel: { color: "#636b65", fontSize: 11 },
        splitLine: { lineStyle: { color: "#e4e8e1" } },
      },
      {
        type: "value",
        name: "增长",
        nameTextStyle: { color: "#636b65", fontSize: 11 },
        axisLabel: {
          color: "#636b65",
          fontSize: 11,
          formatter: (value: number) => `${(value * 100).toFixed(0)}%`,
        },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: "热度",
        type: "line",
        smooth: true,
        symbolSize: 5,
        lineStyle: { width: 2.4 },
        data: ordered.map((point) => point.heat),
        markLine: props.sourceWeek
          ? {
              silent: true,
              symbol: "none",
              label: {
                formatter: "预测窗口",
                color: "#B42318",
                fontSize: 11,
              },
              lineStyle: {
                color: "#B42318",
                type: "dashed",
                width: 1,
              },
              data: [{ xAxis: `W${props.sourceWeek}` }],
            }
          : undefined,
      },
      {
        name: "预测增长",
        type: "line",
        yAxisIndex: 1,
        smooth: true,
        symbolSize: 5,
        lineStyle: { width: 2 },
        data: ordered.map((point) => point.pred_target_growth),
      },
      {
        name: "实际增长",
        type: "line",
        yAxisIndex: 1,
        smooth: true,
        symbolSize: 5,
        lineStyle: { width: 1.6, type: "dashed" },
        data: ordered.map((point) => point.actual_target_growth),
      },
    ],
  };
});

type TooltipParam = {
  dataIndex: number;
  marker: string;
  seriesName: string;
};

function formatTooltip(params: unknown, points: HeatSeriesPoint[]) {
  const items = Array.isArray(params) ? (params as TooltipParam[]) : [];
  const point = points[items[0]?.dataIndex ?? -1];
  if (!point) {
    return "";
  }
  const seriesLines = items
    .map((item) => {
      const value =
        item.seriesName === "热度"
          ? formatNumber(point.heat, 3)
          : item.seriesName === "预测增长"
            ? formatSignedPercent(point.pred_target_growth, 1)
            : formatSignedPercent(point.actual_target_growth, 1);
      return `${item.marker}${item.seriesName}: ${value}`;
    })
    .join("<br/>");
  return [
    `<strong>W${point.week_id}</strong>`,
    seriesLines,
    `pred_share_t1: ${formatSignedPercent(point.pred_share_t1, 1)}`,
  ].join("<br/>");
}
</script>
