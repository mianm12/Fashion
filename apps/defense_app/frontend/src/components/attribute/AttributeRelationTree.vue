<template>
  <div class="attribute-relation-tree">
    <StatusBlock
      v-if="!graph || graph.nodes.length === 0"
      title="暂无属性关系"
      message="当前属性没有可展示的层级边"
      class="compact-state"
    />
    <div v-else class="attribute-tree-canvas">
      <div class="tree-column parent-column">
        <span class="tree-column-title">父级</span>
        <RouterLink
          v-for="node in parentNodes"
          :key="node.id"
          class="tree-node parent-node"
          :to="attributeRoute(node.id)"
        >
          <strong>{{ node.label }}</strong>
          <span>{{ node.type }}</span>
        </RouterLink>
        <span v-if="parentNodes.length === 0" class="tree-empty-label">无父级</span>
      </div>

      <div class="tree-column current-column">
        <span class="tree-column-title">当前属性</span>
        <RouterLink
          v-if="currentNode"
          class="tree-node current-node"
          :to="attributeRoute(currentNode.id)"
        >
          <strong>{{ currentNode.label }}</strong>
          <span>{{ currentNode.type }}</span>
        </RouterLink>
      </div>

      <div class="tree-column child-column">
        <span class="tree-column-title">子级 / 相关</span>
        <RouterLink
          v-for="node in childNodes"
          :key="node.id"
          class="tree-node child-node"
          :to="attributeRoute(node.id)"
        >
          <strong>{{ node.label }}</strong>
          <span>{{ node.type }}</span>
        </RouterLink>
        <span v-if="childNodes.length === 0" class="tree-empty-label">无子级</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

import StatusBlock from "@/components/ui/StatusBlock.vue";
import type { AttributeGraphResponse, GraphNode } from "@/types/api";

const props = defineProps<{
  attrId: string;
  graph: AttributeGraphResponse | null;
  sourceWeek?: number | null;
}>();

const currentNode = computed(() =>
  props.graph?.nodes.find((node) => node.id === props.attrId) ?? null,
);

const parentNodes = computed(() =>
  uniqueNodes(
    props.graph?.edges
      .filter((edge) => edge.target === props.attrId)
      .map((edge) => findNode(edge.source)) ?? [],
  ),
);

const childNodes = computed(() =>
  uniqueNodes(
    props.graph?.edges
      .filter((edge) => edge.source === props.attrId)
      .map((edge) => findNode(edge.target)) ?? [],
  ),
);

function findNode(id: string) {
  return props.graph?.nodes.find((node) => node.id === id) ?? null;
}

function uniqueNodes(nodes: Array<GraphNode | null>) {
  const seen = new Set<string>();
  return nodes.filter((node): node is GraphNode => {
    if (!node || seen.has(node.id)) {
      return false;
    }
    seen.add(node.id);
    return true;
  });
}

function attributeRoute(attrId: string) {
  return {
    name: "attribute-detail",
    params: { attrId },
    query: props.sourceWeek ? { source_week: props.sourceWeek } : {},
  };
}
</script>
