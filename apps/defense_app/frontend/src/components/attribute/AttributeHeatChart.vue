<template>
  <div class="heat-chart-wrap">
    <div v-if="points.length === 0" class="dense-state chart-empty">
      暂无最近 8 周 heat / growth 序列
    </div>
    <VChart v-else class="heat-chart" :option="option" autoresize />
  </div>
</template>

<script setup lang="ts">
import { LineChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from "echarts/components";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import VChart from "vue-echarts";
import { computed } from "vue";

import type { HeatSeriesPoint } from "@/types/api";

use([CanvasRenderer, GridComponent, LegendComponent, LineChart, TooltipComponent]);

const props = defineProps<{
  points: HeatSeriesPoint[];
}>();

const option = computed(() => {
  const ordered = [...props.points].sort((left, right) => left.week_id - right.week_id);

  return {
    color: ["#0f766e", "#b42318", "#9a6700"],
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
      valueFormatter: (value: unknown) =>
        typeof value === "number" ? value.toFixed(4) : String(value),
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
        name: "heat",
        nameTextStyle: { color: "#636b65", fontSize: 11 },
        axisLabel: { color: "#636b65", fontSize: 11 },
        splitLine: { lineStyle: { color: "#e4e8e1" } },
      },
      {
        type: "value",
        name: "growth",
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
        name: "heat",
        type: "line",
        smooth: true,
        symbolSize: 5,
        data: ordered.map((point) => point.heat),
      },
      {
        name: "actual growth",
        type: "line",
        yAxisIndex: 1,
        smooth: true,
        symbolSize: 5,
        data: ordered.map((point) => point.actual_target_growth),
      },
      {
        name: "pred growth",
        type: "line",
        yAxisIndex: 1,
        smooth: true,
        symbolSize: 5,
        data: ordered.map((point) => point.pred_target_growth),
      },
    ],
  };
});
</script>
