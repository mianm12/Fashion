<template>
  <div class="metric-strip" aria-label="关键指标">
    <article
      v-for="metric in visibleMetrics"
      :key="metricKey(metric)"
      class="metric-tile"
    >
      <span class="metric-domain">{{ metric.metric_domain }}</span>
      <strong>{{ formatMetricValue(metric.metric_value) }}</strong>
      <span class="metric-name">
        {{ metric.model_or_method }} / {{ metric.split }}
      </span>
      <span class="metric-label">{{ metric.metric_name }}</span>
    </article>

    <article v-if="visibleMetrics.length === 0" class="metric-tile empty-tile">
      <span class="metric-domain">metrics</span>
      <strong>--</strong>
      <span class="metric-name">暂无摘要指标</span>
      <span class="metric-label">等待后端返回</span>
    </article>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

import type { MetricItem } from "@/types/api";

const props = defineProps<{
  groups: Record<string, MetricItem[]>;
}>();

const visibleMetrics = computed(() =>
  Object.values(props.groups)
    .flat()
    .sort((left, right) => left.display_order - right.display_order)
    .slice(0, 6),
);

function metricKey(metric: MetricItem) {
  return [
    metric.metric_domain,
    metric.model_or_method,
    metric.split,
    metric.metric_name,
  ].join("::");
}

function formatMetricValue(value: number) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 1) {
    return value.toFixed(2);
  }
  return value.toFixed(4);
}
</script>
