<template>
  <div class="graph-canvas-wrap">
    <StatusBlock
      v-if="nodes.length === 0 || edges.length === 0"
      title="暂无商品属性边"
      message="当前商品没有可展示的属性连接"
      class="graph-empty"
    />
    <svg
      v-else
      class="graph-canvas"
      viewBox="0 0 900 520"
      role="img"
      :aria-label="`${article.label} 的属性图`"
    >
      <defs>
        <pattern id="graph-grid" width="36" height="36" patternUnits="userSpaceOnUse">
          <path d="M36 0H0V36" fill="none" stroke="#EEF1ED" stroke-width="1" />
        </pattern>
      </defs>
      <rect width="900" height="520" fill="url(#graph-grid)" />

      <g class="graph-edge-layer">
        <g
          v-for="edge in positionedEdges"
          :key="edge.key"
          :class="edgeClasses(edge)"
        >
          <line
            :x1="edge.source.x"
            :y1="edge.source.y"
            :x2="edge.target.x"
            :y2="edge.target.y"
          />
          <text :x="edge.labelX" :y="edge.labelY">{{ edge.relation }}</text>
        </g>
      </g>

      <g class="graph-node-layer">
        <g
          v-for="node in positionedNodes"
          :key="node.id"
          class="graph-svg-node"
          :class="nodeClasses(node)"
          :transform="`translate(${node.x} ${node.y})`"
          role="button"
          tabindex="0"
          @click="emit('selectNode', node.id)"
          @keydown.enter.prevent="emit('selectNode', node.id)"
        >
          <rect
            :x="-node.width / 2"
            :y="-node.height / 2"
            :width="node.width"
            :height="node.height"
            rx="8"
          />
          <text y="-4">{{ shortText(node.label, node.kind === "article" ? 30 : 22) }}</text>
          <text y="15" class="node-subtitle">{{ graphTypeLabel(node.type) }}</text>
        </g>
      </g>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

import { graphTypeLabel, graphTypeTone } from "@/components/graph/graphTypes";
import StatusBlock from "@/components/ui/StatusBlock.vue";
import type { GraphEdge, GraphNode } from "@/types/api";

const props = withDefaults(
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

type PositionedNode = GraphNode & {
  kind: "article" | "attribute";
  x: number;
  y: number;
  width: number;
  height: number;
  muted: boolean;
  tone: string;
};

type PositionedEdge = {
  key: string;
  relation: string;
  source: PositionedNode;
  target: PositionedNode;
  targetType: string;
  labelX: number;
  labelY: number;
  muted: boolean;
};

const typeAngles: Record<string, number> = {
  colour_group_name: -135,
  product_type_name: -72,
  graphical_appearance_name: -16,
  garment_group_name: 42,
  department_name: 102,
  section_name: 155,
  index_name: 210,
  index_group_name: 265,
};

const positionedNodes = computed<PositionedNode[]>(() => {
  const articleNode: PositionedNode = {
    ...props.article,
    kind: "article",
    type: "article",
    x: 450,
    y: 260,
    width: 238,
    height: 72,
    muted: false,
    tone: "article",
  };
  const groups = groupNodes(props.nodes);
  const attributeNodes = [...groups.entries()].flatMap(([type, nodes]) =>
    nodes.map((node, index) => placeAttributeNode(node, type, index, nodes.length)),
  );
  return [articleNode, ...attributeNodes];
});

const nodeById = computed(() => {
  const map = new Map<string, PositionedNode>();
  for (const node of positionedNodes.value) {
    map.set(node.id, node);
  }
  return map;
});

const positionedEdges = computed<PositionedEdge[]>(() =>
  props.edges.flatMap((edge, index) => {
    const source = nodeById.value.get(edge.source);
    const target = nodeById.value.get(edge.target);
    if (!source || !target) {
      return [];
    }
    const muted = isTypeMuted(target.type);
    return [
      {
        key: `${edge.source}-${edge.target}-${edge.relation_type}-${index}`,
        relation: edge.relation_type,
        source,
        target,
        targetType: target.type,
        labelX: (source.x + target.x) / 2,
        labelY: (source.y + target.y) / 2 - 8,
        muted,
      },
    ];
  }),
);

function groupNodes(nodes: GraphNode[]) {
  const groups = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    const type = node.type || "attribute";
    groups.set(type, [...(groups.get(type) ?? []), node]);
  }
  return new Map([...groups.entries()].sort(([left], [right]) => typeOrder(left) - typeOrder(right)));
}

function placeAttributeNode(
  node: GraphNode,
  type: string,
  index: number,
  groupSize: number,
): PositionedNode {
  const angle = ((typeAngles[type] ?? fallbackAngle(type)) * Math.PI) / 180;
  const centerX = 450 + Math.cos(angle) * 310;
  const centerY = 260 + Math.sin(angle) * 178;
  const spread = Math.min(72, 190 / Math.max(groupSize, 1));
  const offset = (index - (groupSize - 1) / 2) * spread;
  const x = clamp(centerX + Math.cos(angle + Math.PI / 2) * offset, 105, 795);
  const y = clamp(centerY + Math.sin(angle + Math.PI / 2) * offset, 54, 466);
  return {
    ...node,
    kind: "attribute",
    x,
    y,
    width: 176,
    height: 54,
    muted: isTypeMuted(type),
    tone: graphTypeTone(type),
  };
}

function typeOrder(type: string) {
  const keys = Object.keys(typeAngles);
  const index = keys.indexOf(type);
  return index === -1 ? keys.length : index;
}

function fallbackAngle(type: string) {
  return 300 + (hashString(type) % 48);
}

function hashString(value: string) {
  return [...value].reduce((sum, char) => sum + char.charCodeAt(0), 0);
}

function isTypeMuted(type: string) {
  return props.activeType !== "all" && type !== "article" && type !== props.activeType;
}

function nodeClasses(node: PositionedNode) {
  return {
    [`graph-node-${node.tone}`]: true,
    "is-selected": node.id === props.selectedNodeId,
    "is-muted": node.muted,
  };
}

function edgeClasses(edge: PositionedEdge) {
  const selected = props.selectedNodeId
    ? edge.source.id === props.selectedNodeId || edge.target.id === props.selectedNodeId
    : false;
  return {
    "graph-svg-edge": true,
    "is-selected": selected,
    "is-muted": edge.muted,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function shortText(value: string, maxLength: number) {
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}...` : value;
}
</script>
