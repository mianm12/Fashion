<template>
  <Panel title="关系边明细" subtitle="用于审计商品到属性的结构连接">
    <template #actions>
      <span class="count-pill">{{ edges.length }} 条边</span>
    </template>
    <StatusBlock
      v-if="edges.length === 0"
      title="暂无关系边"
      message="当前商品没有可展示的属性连接"
      class="compact-state"
    />
    <div v-else class="graph-edge-table">
      <div class="graph-edge-row graph-edge-head">
        <span>source</span>
        <span>relation_type</span>
        <span>target</span>
        <span>attr_type</span>
        <span>attr_value</span>
        <span>操作</span>
      </div>
      <div
        v-for="row in rows"
        :key="row.key"
        class="graph-edge-row"
        :class="{ muted: isMuted(row.attr_type) }"
      >
        <span>{{ row.source }}</span>
        <code>{{ row.relation_type }}</code>
        <span>{{ row.target }}</span>
        <span>{{ graphTypeLabel(row.attr_type) }}</span>
        <strong>{{ row.attr_value }}</strong>
        <RouterLink :to="attributeRoute(row.target)">查看属性详情</RouterLink>
      </div>
    </div>
  </Panel>
</template>

<script setup lang="ts">
import { computed } from "vue";

import { graphTypeLabel } from "@/components/graph/graphTypes";
import Panel from "@/components/ui/Panel.vue";
import StatusBlock from "@/components/ui/StatusBlock.vue";
import type { GraphEdge, GraphNode } from "@/types/api";

const props = withDefaults(
  defineProps<{
    nodes: GraphNode[];
    edges: GraphEdge[];
    activeType?: string;
    sourceWeek?: number | null;
  }>(),
  {
    activeType: "all",
    sourceWeek: null,
  },
);

const nodeById = computed(() => new Map(props.nodes.map((node) => [node.id, node])));

const rows = computed(() =>
  props.edges.map((edge, index) => {
    const target = nodeById.value.get(edge.target);
    return {
      key: `${edge.source}-${edge.target}-${edge.relation_type}-${index}`,
      source: edge.source,
      relation_type: edge.relation_type,
      target: edge.target,
      attr_type: target?.type ?? "--",
      attr_value: target?.label ?? "--",
    };
  }),
);

function isMuted(attrType: string) {
  return props.activeType !== "all" && attrType !== props.activeType;
}

function attributeRoute(attrId: string) {
  return {
    name: "attribute-detail",
    params: { attrId },
    query: props.sourceWeek ? { source_week: props.sourceWeek } : {},
  };
}
</script>
