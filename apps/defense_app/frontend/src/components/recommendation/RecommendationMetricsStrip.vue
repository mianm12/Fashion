<template>
  <section class="recommendation-metrics-strip">
    <StatusBlock
      v-if="loading"
      type="loading"
      title="加载推荐指标..."
      class="compact-state"
    />
    <StatusBlock
      v-else-if="error"
      type="error"
      title="推荐指标加载失败"
      :message="error"
      class="compact-state"
    />
    <template v-else>
      <MetricTile label="推荐方法" value="pop_similarity_trend" hint="稳定结果" tone="blue">
        <template #icon>
          <GitBranch aria-hidden="true" />
        </template>
      </MetricTile>
      <MetricTile
        v-for="tile in metricTiles"
        :key="tile.name"
        :label="tile.label"
        :value="tile.value"
        :hint="tile.hint"
        :tone="tile.tone"
      >
        <template #icon>
          <component :is="tile.icon" aria-hidden="true" />
        </template>
      </MetricTile>
    </template>
  </section>
</template>

<script setup lang="ts">
import { BadgeCheck, GitBranch, MousePointerClick, Target } from "lucide-vue-next";
import { computed } from "vue";

import MetricTile from "@/components/ui/MetricTile.vue";
import StatusBlock from "@/components/ui/StatusBlock.vue";
import type { MetricItem } from "@/types/api";
import { formatPercent } from "@/utils/formatters";

const props = defineProps<{
  metrics: MetricItem[];
  loading?: boolean;
  error?: string | null;
}>();

const metricConfig = [
  { name: "map_at_12", label: "MAP@12", icon: Target, tone: "red" },
  { name: "recall_at_12", label: "Recall@12", icon: MousePointerClick, tone: "green" },
  { name: "hit_rate_at_12", label: "Hit Rate@12", icon: BadgeCheck, tone: "gold" },
] as const;

const metricTiles = computed(() =>
  metricConfig.flatMap((config) => {
    const metric = props.metrics.find(
      (item) =>
        item.model_or_method === "pop_similarity_trend" &&
        item.split === "test" &&
        item.metric_name === config.name,
    );
    if (!metric) {
      return [];
    }
    return [
      {
        ...config,
        value: formatPercent(metric.metric_value, 2),
        hint: "test 切分",
      },
    ];
  }),
);
</script>
