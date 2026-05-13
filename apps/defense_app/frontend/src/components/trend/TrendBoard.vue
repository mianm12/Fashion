<template>
  <article class="trend-board panel">
    <header class="panel-heading">
      <div>
        <h2>{{ title }}</h2>
        <p>{{ attrType }}</p>
      </div>
      <span class="count-pill">{{ items.length }} rows</span>
    </header>

    <div v-if="loading" class="dense-state compact-state">
      加载趋势榜...
    </div>

    <div v-else-if="error" class="dense-state error-state compact-state">
      {{ error }}
    </div>

    <div v-else-if="items.length === 0" class="dense-state compact-state">
      暂无趋势属性
    </div>

    <ol v-else class="trend-list">
      <li v-for="item in items" :key="item.attr_id" class="trend-row">
        <span class="rank-cell">#{{ item.rank }}</span>
        <RouterLink
          class="trend-attr-link"
          :to="{ name: 'attribute-detail', params: { attrId: item.attr_id } }"
        >
          <strong>{{ item.attr_value }}</strong>
          <span>{{ item.attr_id }}</span>
        </RouterLink>
        <span class="trend-value">
          {{ formatSignedPercent(item.pred_target_growth) }}
        </span>
      </li>
    </ol>
  </article>
</template>

<script setup lang="ts">
import type { TrendAttribute } from "@/types/api";

defineProps<{
  title: string;
  attrType: string;
  items: TrendAttribute[];
  loading?: boolean;
  error?: string | null;
}>();

function formatSignedPercent(value: number | null) {
  if (value === null || !Number.isFinite(value)) {
    return "--";
  }
  const formatted = `${(value * 100).toFixed(1)}%`;
  return value > 0 ? `+${formatted}` : formatted;
}
</script>
