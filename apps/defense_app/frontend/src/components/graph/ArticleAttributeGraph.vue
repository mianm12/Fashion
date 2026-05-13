<template>
  <GraphCanvas
    :article="article"
    :nodes="nodes"
    :edges="edges"
    :selected-node-id="selectedNodeId"
    :active-type="activeType"
    @select-node="(nodeId) => emit('selectNode', nodeId)"
  />
</template>

<script setup lang="ts">
import GraphCanvas from "@/components/graph/GraphCanvas.vue";
import type { GraphEdge, GraphNode } from "@/types/api";

withDefaults(
  defineProps<{
    article: GraphNode;
    nodes: GraphNode[];
    edges: GraphEdge[];
    selectedNodeId?: string | null;
    activeType?: string;
  }>(),
  {
    selectedNodeId: null,
    activeType: "all",
  },
);

const emit = defineEmits<{
  selectNode: [nodeId: string];
}>();
</script>
