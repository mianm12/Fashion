<template>
  <div class="metric-strip" aria-label="关键指标">
    <StatusBlock
      v-if="loading"
      class="metric-strip-state"
      type="loading"
      title="加载趋势指标..."
    />
    <StatusBlock
      v-else-if="error"
      class="metric-strip-state"
      type="error"
      title="趋势指标加载失败"
      :message="error"
    />
    <template v-else>
      <MetricTile
        label="上升属性总数"
        :value="formatInteger(summary?.rising_attribute_count)"
        hint="核心四类 Top-K"
        tone="red"
      >
        <template #icon><TrendingUp /></template>
      </MetricTile>
      <MetricTile
        label="高置信属性数"
        :value="formatInteger(summary?.high_confidence_attribute_count)"
        hint="预测增长 >= 20%"
        tone="green"
      >
        <template #icon><BarChart3 /></template>
      </MetricTile>
      <MetricTile
        label="Top-K 平均预测分"
        :value="formatNumber(summary?.top_k_average_pred_target_growth)"
        hint="预测增长均值"
        tone="gold"
      >
        <template #icon><Star /></template>
      </MetricTile>
      <MetricTile
        label="覆盖商品数"
        :value="formatInteger(summary?.covered_article_count)"
        hint="去重商品 ID"
        tone="blue"
      >
        <template #icon><PackageCheck /></template>
      </MetricTile>
      <MetricTile
        label="模型状态"
        :value="summary?.model_status ?? '--'"
        hint="稳定产物"
        tone="neutral"
      >
        <template #icon><Activity /></template>
      </MetricTile>
    </template>
  </div>
</template>

<script setup lang="ts">
import {
  Activity,
  BarChart3,
  PackageCheck,
  Star,
  TrendingUp,
} from "lucide-vue-next";

import MetricTile from "@/components/ui/MetricTile.vue";
import StatusBlock from "@/components/ui/StatusBlock.vue";
import { formatInteger, formatNumber } from "@/utils/formatters";
import type { TrendSummaryResponse } from "@/types/api";

defineProps<{
  summary: TrendSummaryResponse | null;
  loading?: boolean;
  error?: string | null;
}>();
</script>
