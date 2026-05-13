<template>
  <div class="score-breakdown">
    <div v-if="!scores" class="dense-state compact-state">
      选择推荐商品查看 score breakdown
    </div>
    <template v-else>
      <div v-for="item in scoreItems" :key="item.key" class="score-row">
        <div>
          <strong>{{ item.label }}</strong>
          <span>{{ formatScore(item.value) }}</span>
        </div>
        <div class="score-track" aria-hidden="true">
          <span class="score-fill" :style="{ width: item.width }" />
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

import type { ScoreComponents } from "@/types/api";

const props = defineProps<{
  scores: ScoreComponents | null;
}>();

const scoreItems = computed(() => {
  if (!props.scores) {
    return [];
  }
  const items = [
    { key: "pop_score", label: "pop_score", value: props.scores.pop_score },
    { key: "sim_score", label: "sim_score", value: props.scores.sim_score },
    { key: "trend_score", label: "trend_score", value: props.scores.trend_score },
    { key: "recent_score", label: "recent_score", value: props.scores.recent_score },
    { key: "final_score", label: "final_score", value: props.scores.final_score },
  ];
  const max = Math.max(...items.map((item) => Math.abs(item.value)), 1);
  return items.map((item) => ({
    ...item,
    width: `${Math.max((Math.abs(item.value) / max) * 100, 4).toFixed(1)}%`,
  }));
});

function formatScore(value: number) {
  return Number.isFinite(value) ? value.toFixed(4) : "--";
}
</script>
