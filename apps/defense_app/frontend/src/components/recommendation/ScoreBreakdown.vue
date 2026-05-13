<template>
  <div class="score-breakdown">
    <div v-if="!scores" class="dense-state compact-state">
      选择推荐商品查看分数分解
    </div>
    <template v-else>
      <div
        v-for="item in scoreItems"
        :key="item.key"
        class="score-row"
        :class="`score-row-${item.key}`"
      >
        <div>
          <strong>{{ item.label }}</strong>
          <small>{{ item.hint }}</small>
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
    { key: "final_score", label: "最终分", hint: "排序分", value: props.scores.final_score },
    { key: "pop_score", label: "流行度", value: props.scores.pop_score },
    { key: "sim_score", label: "相似度", hint: "用户偏好", value: props.scores.sim_score },
    { key: "trend_score", label: "趋势分", hint: "趋势属性", value: props.scores.trend_score },
    { key: "recent_score", label: "近期热度", hint: "近窗热度", value: props.scores.recent_score },
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
