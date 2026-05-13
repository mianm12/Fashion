<template>
  <div class="graph-inspector-stack">
    <Panel title="节点详情" subtitle="点击画布节点查看结构证据">
      <StatusBlock
        v-if="!selectedNode"
        title="未选择节点"
        message="点击商品或属性节点查看详情"
        class="compact-state"
      />
      <dl v-else class="node-inspector">
        <div>
          <dt>节点类型</dt>
          <dd>{{ graphTypeLabel(selectedNode.type) }}</dd>
        </div>
        <div>
          <dt>节点标签</dt>
          <dd>{{ selectedNode.label }}</dd>
        </div>
        <div>
          <dt>节点 ID</dt>
          <dd>{{ selectedNode.id }}</dd>
        </div>
        <div>
          <dt>关联边数</dt>
          <dd>{{ selectedRelationCount }}</dd>
        </div>
        <div>
          <dt>相邻节点数</dt>
          <dd>{{ selectedNeighborCount }}</dd>
        </div>
      </dl>
      <div v-if="selectedNode" class="node-actions">
        <RouterLink
          v-if="selectedNode.type !== 'article'"
          class="panel-action-link"
          :to="attributeRoute(selectedNode.id)"
        >
          查看属性详情
        </RouterLink>
        <button type="button" class="mini-action-button" @click="copyNodeId">
          {{ copyState }}
        </button>
      </div>
    </Panel>

    <Panel title="属性分组" subtitle="按属性类型统计当前商品节点">
      <div class="graph-group-list">
        <button
          v-for="group in groupRows"
          :key="group.type"
          type="button"
          :class="{ active: activeType === group.type }"
          @click="emit('selectType', group.type)"
        >
          <span>{{ graphTypeLabel(group.type) }}</span>
          <strong>{{ group.count }}</strong>
        </button>
      </div>
    </Panel>

    <Panel title="关系说明" subtitle="只读展示图结构">
      <p class="relation-note">
        <code>has_attribute</code> 表示商品拥有某个属性值；点击属性节点可回到属性详情页。
      </p>
    </Panel>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";

import { graphTypeLabel } from "@/components/graph/graphTypes";
import Panel from "@/components/ui/Panel.vue";
import StatusBlock from "@/components/ui/StatusBlock.vue";
import type { GraphEdge, GraphNode } from "@/types/api";

const props = withDefaults(
  defineProps<{
    article: GraphNode | null;
    nodes: GraphNode[];
    edges: GraphEdge[];
    selectedNodeId?: string | null;
    activeType?: string;
    sourceWeek?: number | null;
  }>(),
  {
    selectedNodeId: null,
    activeType: "all",
    sourceWeek: null,
  },
);

const emit = defineEmits<{
  selectType: [type: string];
}>();

const copyState = ref("复制 ID");

const allNodes = computed(() =>
  props.article ? [props.article, ...props.nodes] : props.nodes,
);

const selectedNode = computed(() =>
  allNodes.value.find((node) => node.id === props.selectedNodeId) ?? null,
);

const selectedEdges = computed(() =>
  selectedNode.value
    ? props.edges.filter(
        (edge) =>
          edge.source === selectedNode.value?.id || edge.target === selectedNode.value?.id,
      )
    : [],
);

const selectedRelationCount = computed(() => selectedEdges.value.length);
const selectedNeighborCount = computed(
  () =>
    new Set(
      selectedEdges.value.flatMap((edge) => [edge.source, edge.target]).filter(
        (id) => id !== selectedNode.value?.id,
      ),
    ).size,
);

const groupRows = computed(() => {
  const counts = new Map<string, number>();
  for (const node of props.nodes) {
    counts.set(node.type, (counts.get(node.type) ?? 0) + 1);
  }
  return [
    { type: "all", count: props.nodes.length },
    ...[...counts.entries()].map(([type, count]) => ({ type, count })),
  ];
});

watch(
  () => props.selectedNodeId,
  () => {
    copyState.value = "复制 ID";
  },
);

async function copyNodeId() {
  if (!selectedNode.value) {
    return;
  }
  try {
    await navigator.clipboard.writeText(selectedNode.value.id);
    copyState.value = "已复制";
  } catch {
    copyState.value = "复制失败";
  }
}

function attributeRoute(attrId: string) {
  return {
    name: "attribute-detail",
    params: { attrId },
    query: props.sourceWeek ? { source_week: props.sourceWeek } : {},
  };
}
</script>
