<template>
  <article class="trend-board panel">
    <header class="panel-heading">
      <div>
        <h2>{{ title }}</h2>
        <p>{{ attrType }}</p>
      </div>
      <span class="count-pill">{{ items.length }} 条</span>
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
          :to="{
            name: 'attribute-detail',
            params: { attrId: item.attr_id },
            query: sourceWeek ? { source_week: sourceWeek } : {},
          }"
        >
          <strong>{{ item.attr_value }}</strong>
          <span>{{ item.attr_id }}</span>
        </RouterLink>
        <div class="trend-strength">
          <span class="trend-bar-track" aria-hidden="true">
            <span
              class="trend-bar-fill"
              :style="{ width: strengthWidth(item.pred_target_growth) }"
            />
          </span>
          <span class="trend-value">
            {{ formatSignedPercent(item.pred_target_growth) }}
          </span>
        </div>
      </li>
    </ol>
  </article>
</template>

<script setup lang="ts">
import { computed } from "vue";

import type { TrendAttribute } from "@/types/api";
import { formatSignedPercent } from "@/utils/formatters";

const props = defineProps<{
  title: string;
  attrType: string;
  items: TrendAttribute[];
  sourceWeek?: number | null;
  loading?: boolean;
  error?: string | null;
}>();

const maxGrowth = computed(() =>
  Math.max(
    ...props.items.map((item) => Math.abs(item.pred_target_growth ?? 0)),
    0.01,
  ),
);

function strengthWidth(value: number | null) {
  if (value === null || !Number.isFinite(value)) {
    return "0%";
  }
  return `${Math.max((Math.abs(value) / maxGrowth.value) * 100, 6).toFixed(1)}%`;
}
</script>
