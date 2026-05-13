<template>
  <div class="ui-table-wrap">
    <table class="ui-table">
      <thead>
        <tr>
          <th
            v-for="column in columns"
            :key="column.key"
            :class="column.align ? `align-${column.align}` : undefined"
          >
            {{ column.label }}
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-if="rows.length === 0">
          <td :colspan="columns.length" class="ui-table-empty">
            {{ emptyText }}
          </td>
        </tr>
        <tr v-for="row in rows" v-else :key="rowKeyValue(row)">
          <td
            v-for="column in columns"
            :key="column.key"
            :class="column.align ? `align-${column.align}` : undefined"
          >
            <slot :name="column.key" :row="row" :value="row[column.key]">
              {{ row[column.key] ?? "--" }}
            </slot>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
export interface DataTableColumn {
  key: string;
  label: string;
  align?: "left" | "center" | "right";
}

const props = withDefaults(
  defineProps<{
    columns: DataTableColumn[];
    rows: Array<Record<string, unknown>>;
    rowKey?: string;
    emptyText?: string;
  }>(),
  {
    rowKey: "id",
    emptyText: "暂无数据",
  },
);

function rowKeyValue(row: Record<string, unknown>) {
  return String(row[props.rowKey] ?? JSON.stringify(row));
}
</script>
