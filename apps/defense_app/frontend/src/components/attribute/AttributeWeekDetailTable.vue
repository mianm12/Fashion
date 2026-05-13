<template>
  <div class="attribute-week-table">
    <StatusBlock
      v-if="points.length === 0"
      title="暂无周级明细"
      message="当前属性没有可展示的热度序列"
      class="compact-state"
    />
    <table v-else>
      <thead>
        <tr>
          <th>周</th>
          <th>热度</th>
          <th>预测增长</th>
          <th>真实增长</th>
          <th>预测份额</th>
          <th>预测方向</th>
          <th>真实方向</th>
          <th>方向一致性</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="row in tableRows"
          :key="row.week_id"
          :class="{ 'current-week-row': row.week_id === sourceWeek }"
        >
          <td>{{ row.week_id }}</td>
          <td>{{ formatNumber(row.heat, 3) }}</td>
          <td>{{ formatSignedPercent(row.pred_target_growth, 1) }}</td>
          <td>{{ formatSignedPercent(row.actual_target_growth, 1) }}</td>
          <td>{{ formatSignedPercent(row.pred_share_t1, 1) }}</td>
          <td>{{ directionLabel(row.pred_target_growth) }}</td>
          <td>{{ directionLabel(row.actual_target_growth) }}</td>
          <td>
            <span :class="matchClass(row)">
              {{ matchLabel(row) }}
            </span>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

import StatusBlock from "@/components/ui/StatusBlock.vue";
import type { HeatSeriesPoint } from "@/types/api";
import { formatNumber, formatSignedPercent } from "@/utils/formatters";

const props = defineProps<{
  points: HeatSeriesPoint[];
  sourceWeek?: number | null;
}>();

const tableRows = computed(() =>
  [...props.points].sort((left, right) => left.week_id - right.week_id),
);

function directionLabel(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  if (value > 0) {
    return "上升";
  }
  if (value < 0) {
    return "下降";
  }
  return "持平";
}

function directionSign(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  if (value > 0) {
    return 1;
  }
  if (value < 0) {
    return -1;
  }
  return 0;
}

function matchLabel(row: HeatSeriesPoint) {
  const pred = directionSign(row.pred_target_growth);
  const actual = directionSign(row.actual_target_growth);
  if (pred === null || actual === null) {
    return "--";
  }
  return pred === actual ? "一致" : "偏离";
}

function matchClass(row: HeatSeriesPoint) {
  const label = matchLabel(row);
  return {
    "match-chip": true,
    "match-chip-ok": label === "一致",
    "match-chip-miss": label === "偏离",
  };
}
</script>
