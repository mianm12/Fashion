<template>
  <section class="trend-evidence-grid" aria-label="趋势证据">
    <article class="panel trend-distribution-panel">
      <header class="panel-heading">
        <div>
          <h2>属性趋势分布</h2>
          <p>按预测增长区间统计核心属性</p>
        </div>
      </header>
      <StatusBlock
        v-if="loading"
        type="loading"
        title="加载分布..."
        class="compact-state"
      />
      <StatusBlock
        v-else-if="error"
        type="error"
        title="趋势证据加载失败"
        :message="error"
        class="compact-state"
      />
      <VChart
        v-else-if="evidence?.distribution.length"
        class="trend-evidence-chart"
        :option="distributionOption"
        autoresize
      />
      <StatusBlock v-else title="暂无分布数据" class="compact-state" />
    </article>

    <article class="panel trend-history-panel">
      <header class="panel-heading">
        <div>
          <h2>趋势走势</h2>
          <p>各维度 Top-1 属性最近周预测增长</p>
        </div>
      </header>
      <StatusBlock
        v-if="loading"
        type="loading"
        title="加载走势..."
        class="compact-state"
      />
      <StatusBlock
        v-else-if="error"
        type="error"
        title="趋势走势加载失败"
        :message="error"
        class="compact-state"
      />
      <VChart
        v-else-if="evidence?.top_history.length"
        class="trend-evidence-chart"
        :option="historyOption"
        autoresize
      />
      <StatusBlock v-else title="暂无走势数据" class="compact-state" />
    </article>

    <article class="panel trend-new-panel">
      <header class="panel-heading">
        <div>
          <h2>近期新增高潜属性</h2>
          <p>首次进入当前 Top-K 的属性</p>
        </div>
      </header>
      <StatusBlock
        v-if="loading"
        type="loading"
        title="加载新增属性..."
        class="compact-state"
      />
      <StatusBlock
        v-else-if="error"
        type="error"
        title="新增属性加载失败"
        :message="error"
        class="compact-state"
      />
      <div v-else-if="evidence?.new_high_potential.length" class="new-trend-list">
        <RouterLink
          v-for="item in evidence.new_high_potential"
          :key="`${item.source_week}-${item.attr_id}`"
          :to="{
            name: 'attribute-detail',
            params: { attrId: item.attr_id },
            query: { source_week: item.source_week },
          }"
        >
          <strong>{{ item.attr_value }}</strong>
          <span>{{ labelForAttrType(item.attr_type) }}</span>
          <span>{{ formatSignedPercent(item.pred_target_growth) }}</span>
          <em>新进</em>
        </RouterLink>
      </div>
      <StatusBlock v-else title="暂无新增高潜属性" class="compact-state" />
    </article>
  </section>
</template>

<script setup lang="ts">
import { BarChart, LineChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from "echarts/components";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import VChart from "vue-echarts";
import { computed } from "vue";

import StatusBlock from "@/components/ui/StatusBlock.vue";
import {
  attrTypeLabels,
  type CoreTrendAttrType,
} from "@/components/trend/trendTypes";
import { formatSignedPercent } from "@/utils/formatters";
import type { TrendEvidenceResponse } from "@/types/api";

use([BarChart, CanvasRenderer, GridComponent, LegendComponent, LineChart, TooltipComponent]);

const props = defineProps<{
  evidence: TrendEvidenceResponse | null;
  loading?: boolean;
  error?: string | null;
}>();

const distributionOption = computed(() => ({
  color: ["#3A5A6A"],
  grid: { left: 36, right: 12, top: 20, bottom: 32 },
  tooltip: { trigger: "axis" },
  xAxis: {
    type: "category",
    data: props.evidence?.distribution.map((item) => item.label) ?? [],
    axisLabel: { color: "#66706A", fontSize: 11 },
    axisTick: { show: false },
  },
  yAxis: {
    type: "value",
    axisLabel: { color: "#66706A", fontSize: 11 },
    splitLine: { lineStyle: { color: "#EEF1ED" } },
  },
  series: [
    {
      name: "属性数量",
      type: "bar",
      barWidth: 28,
      data: props.evidence?.distribution.map((item) => item.count) ?? [],
    },
  ],
}));

const historyWeeks = computed(() => [
  ...new Set((props.evidence?.top_history ?? []).map((point) => point.week_id)),
].sort((left, right) => left - right));

const historySeries = computed(() => {
  const points = props.evidence?.top_history ?? [];
  const attrIds = [...new Set(points.map((point) => point.attr_id))];
  return attrIds.map((attrId) => {
    const attrPoints = points.filter((point) => point.attr_id === attrId);
    const first = attrPoints[0];
    return {
      name: first?.attr_value ?? attrId,
      type: "line",
      smooth: true,
      symbolSize: 5,
      data: historyWeeks.value.map((week) => {
        const point = attrPoints.find((item) => item.week_id === week);
        return point?.pred_target_growth ?? null;
      }),
    };
  });
});

const historyOption = computed(() => ({
  color: ["#B42318", "#0F766E", "#9A6700", "#3A5A6A"],
  grid: { left: 42, right: 20, top: 34, bottom: 32 },
  legend: {
    top: 0,
    itemWidth: 14,
    itemHeight: 8,
    textStyle: { color: "#66706A", fontSize: 11 },
  },
  tooltip: {
    trigger: "axis",
    valueFormatter: (value: unknown) =>
      typeof value === "number" ? formatSignedPercent(value) : String(value),
  },
  xAxis: {
    type: "category",
    data: historyWeeks.value.map((week) => `W${week}`),
    boundaryGap: false,
    axisLabel: { color: "#66706A", fontSize: 11 },
    axisTick: { show: false },
  },
  yAxis: {
    type: "value",
    axisLabel: {
      color: "#66706A",
      fontSize: 11,
      formatter: (value: number) => `${(value * 100).toFixed(0)}%`,
    },
    splitLine: { lineStyle: { color: "#EEF1ED" } },
  },
  series: historySeries.value,
}));

function labelForAttrType(attrType: string) {
  return attrTypeLabels[attrType as CoreTrendAttrType] ?? attrType;
}
</script>
