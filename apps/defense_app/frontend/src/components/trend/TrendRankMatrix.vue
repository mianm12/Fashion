<template>
  <div class="trend-rank-matrix" :class="{ 'single-column': attrType !== 'all' }">
    <TrendBoard
      v-for="type in visibleAttrTypes"
      :key="type"
      :title="attrTypeLabels[type]"
      :attr-type="type"
      :items="groups[type] ?? []"
      :source-week="sourceWeek"
      :loading="loading"
      :error="errors[type]"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

import TrendBoard from "@/components/trend/TrendBoard.vue";
import {
  attrTypeLabels,
  coreTrendAttrTypes,
  type CoreTrendAttrType,
  type TrendAttrFilter,
} from "@/components/trend/trendTypes";
import type { TrendAttribute } from "@/types/api";

const props = withDefaults(
  defineProps<{
    groups: Partial<Record<CoreTrendAttrType, TrendAttribute[]>>;
    attrType: TrendAttrFilter;
    sourceWeek?: number | null;
    loading?: boolean;
    errors?: Partial<Record<CoreTrendAttrType, string | null>>;
  }>(),
  {
    errors: () => ({}),
  },
);

const visibleAttrTypes = computed(() =>
  props.attrType === "all"
    ? coreTrendAttrTypes
    : coreTrendAttrTypes.filter((type) => type === props.attrType),
);
</script>
