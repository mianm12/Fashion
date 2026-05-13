<template>
  <div class="article-graph-panel">
    <div v-if="nodes.length === 0 || edges.length === 0" class="dense-state graph-empty">
      暂无商品属性边
    </div>

    <svg
      v-else
      class="article-attribute-graph"
      :viewBox="viewBox"
      role="img"
      :aria-label="`${article.label} 的属性图`"
    >
      <defs>
        <marker
          id="edge-arrow"
          markerHeight="8"
          markerWidth="8"
          orient="auto"
          refX="7"
          refY="4"
          viewBox="0 0 8 8"
        >
          <path d="M0,0 L8,4 L0,8 Z" fill="#8b9490" />
        </marker>
      </defs>

      <g class="edge-layer">
        <g v-for="edge in positionedEdges" :key="edge.key">
          <line
            :x1="edge.source.x"
            :y1="edge.source.y"
            :x2="edge.target.x"
            :y2="edge.target.y"
            marker-end="url(#edge-arrow)"
          />
          <text :x="edge.labelX" :y="edge.labelY">{{ edge.relation }}</text>
        </g>
      </g>

      <g class="node-layer">
        <g
          class="svg-node article-svg-node"
          :transform="`translate(${layout.article.x} ${layout.article.y})`"
        >
          <rect
            :x="-layout.articleNodeWidth / 2"
            y="-34"
            :width="layout.articleNodeWidth"
            height="68"
            rx="8"
          />
          <text y="-4">{{ shortText(article.label, 28) }}</text>
          <text y="16" class="node-subtitle">article</text>
        </g>

        <g
          v-for="node in positionedNodes"
          :key="node.id"
          class="svg-node attr-svg-node"
          :transform="`translate(${node.x} ${node.y})`"
        >
          <rect
            :x="-layout.attrNodeWidth / 2"
            y="-30"
            :width="layout.attrNodeWidth"
            height="60"
            rx="8"
          />
          <text y="-4">{{ node.displayLabel }}</text>
          <text y="15" class="node-subtitle">
            {{ shortText(groupLabel(node.type), 24) }}
          </text>
        </g>
      </g>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

import type { GraphEdge, GraphNode } from "@/types/api";

const props = defineProps<{
  article: GraphNode;
  nodes: GraphNode[];
  edges: GraphEdge[];
}>();

interface PositionedNode extends GraphNode {
  displayLabel: string;
  x: number;
  y: number;
}

const isCompact = ref(false);

const layout = computed(() =>
  isCompact.value
    ? {
        article: { x: 260, y: 260 },
        articleNodeWidth: 210,
        attrNodeWidth: 180,
        height: 520,
        radiusX: 170,
        radiusY: 180,
        width: 520,
      }
    : {
        article: { x: 480, y: 260 },
        articleNodeWidth: 236,
        attrNodeWidth: 204,
        height: 520,
        radiusX: 330,
        radiusY: 180,
        width: 960,
      },
);

const viewBox = computed(() => `0 0 ${layout.value.width} ${layout.value.height}`);

onMounted(() => {
  updateCompactLayout();
  window.addEventListener("resize", updateCompactLayout);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", updateCompactLayout);
});

const positionedNodes = computed<PositionedNode[]>(() => {
  const groups = new Map<string, GraphNode[]>();
  for (const node of props.nodes.filter((item) => item.id !== props.article.id)) {
    const group = node.type || "attribute";
    groups.set(group, [...(groups.get(group) ?? []), node]);
  }

  const groupEntries = [...groups.entries()];

  return groupEntries.flatMap(([group, nodes], groupIndex) => {
    const angle = (Math.PI * 2 * groupIndex) / Math.max(groupEntries.length, 1);
    const centerX = layout.value.article.x + Math.cos(angle) * layout.value.radiusX;
    const centerY = layout.value.article.y + Math.sin(angle) * layout.value.radiusY;
    const nodeGap = 76;
    const offsetStart = -((nodes.length - 1) * nodeGap) / 2;

    return nodes.map((node, nodeIndex) => ({
      ...node,
      type: group,
      displayLabel: shortText(node.label, 26),
      x: clamp(centerX, 102, layout.value.width - 102),
      y: clamp(centerY + offsetStart + nodeIndex * nodeGap, 54, layout.value.height - 54),
    }));
  });
});

const nodePositions = computed(() => {
  const positions = new Map<string, PositionedNode>();
  positions.set(props.article.id, {
    ...props.article,
    displayLabel: shortText(props.article.label, 28),
    ...layout.value.article,
  });
  for (const node of positionedNodes.value) {
    positions.set(node.id, node);
  }
  return positions;
});

const positionedEdges = computed(() =>
  props.edges.flatMap((edge, index) => {
    const source = nodePositions.value.get(edge.source);
    const target = nodePositions.value.get(edge.target);
    if (!source || !target) {
      return [];
    }
    return [
      {
        key: `${edge.source}-${edge.target}-${edge.relation_type}-${index}`,
        source,
        target,
        relation: edge.relation_type,
        labelX: (source.x + target.x) / 2,
        labelY: (source.y + target.y) / 2 - 8,
      },
    ];
  }),
);

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function groupLabel(type: string) {
  return type.replace(/_/g, " ");
}

function shortText(value: string, maxLength: number) {
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}...` : value;
}

function updateCompactLayout() {
  isCompact.value = window.matchMedia("(max-width: 640px)").matches;
}
</script>
