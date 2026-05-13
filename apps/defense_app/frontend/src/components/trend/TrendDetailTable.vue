<template>
  <article class="panel trend-detail-panel">
    <header class="panel-heading">
      <div>
        <h2>趋势属性明细</h2>
        <p>用于答辩追问时展开具体属性行</p>
      </div>
      <span class="count-pill">{{ response?.items.length ?? 0 }} 条</span>
    </header>

    <StatusBlock
      v-if="loading"
      type="loading"
      title="加载趋势明细..."
      class="compact-state"
    />
    <StatusBlock
      v-else-if="error"
      type="error"
      title="趋势明细加载失败"
      :message="error"
      class="compact-state"
    />
    <div v-else-if="!response || response.items.length === 0" class="dense-state compact-state">
      暂无趋势明细
    </div>
    <div v-else class="trend-detail-table">
      <RouterLink
        v-for="item in response.items"
        :key="`${item.source_week}-${item.attr_id}-${item.rank}`"
        class="trend-detail-row"
        :to="{
          name: 'attribute-detail',
          params: { attrId: item.attr_id },
          query: { source_week: item.source_week },
        }"
      >
        <span>#{{ item.rank }}</span>
        <strong>{{ item.attr_value }}</strong>
        <span>{{ item.attr_type }}</span>
        <span>{{ formatNumber(item.heat_t, 1) }}</span>
        <span>{{ formatPercent(item.pred_share_t1) }}</span>
        <span class="trend-value">{{ formatSignedPercent(item.pred_target_growth) }}</span>
        <em>{{ item.is_trend_eligible_t ? "强上升" : "观察" }}</em>
      </RouterLink>
    </div>
  </article>
</template>

<script setup lang="ts">
import StatusBlock from "@/components/ui/StatusBlock.vue";
import {
  formatNumber,
  formatPercent,
  formatSignedPercent,
} from "@/utils/formatters";
import type { TrendListResponse } from "@/types/api";

defineProps<{
  response: TrendListResponse | null;
  loading?: boolean;
  error?: string | null;
}>();
</script>
