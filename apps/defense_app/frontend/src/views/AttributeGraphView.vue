<template>
  <section class="route-page attribute-graph-page">
    <PageToolbar title="属性图展示" :context="toolbarContext" status="只读图谱">
      <template #actions>
        <form class="graph-toolbar-search" @submit.prevent="runSearch">
          <input
            v-model="searchTerm"
            type="search"
            aria-label="商品搜索"
            placeholder="商品 ID / 商品名称"
          />
          <button type="submit" class="toolbar-button">
            <Search aria-hidden="true" />
            搜索
          </button>
        </form>
        <button type="button" class="toolbar-button" @click="resetView">
          <RotateCcw aria-hidden="true" />
          重置视图
        </button>
        <label class="compact-field graph-type-field">
          <span>节点类型</span>
          <select v-model="activeType">
            <option
              v-for="item in graphTypeFilters"
              :key="item.value"
              :value="item.value"
            >
              {{ item.label }}
            </option>
          </select>
        </label>
      </template>
    </PageToolbar>

    <StatusBlock
      v-if="representativeLoading"
      type="loading"
      title="正在选择代表商品..."
      :message="representativeMessage"
      class="compact-state"
    />
    <StatusBlock
      v-else-if="representativeError"
      type="empty"
      title="暂无代表商品"
      :message="representativeError"
      class="compact-state"
    />

    <div class="graph-workspace graph-workspace-three">
      <aside class="graph-sidebar">
        <Panel title="商品搜索结果" subtitle="选择商品查看属性连接">
          <template #actions>
            <span class="count-pill">{{ searchResults.length }} 条结果</span>
          </template>
          <StatusBlock
            v-if="searchLoading"
            type="loading"
            title="搜索商品..."
            class="compact-state"
          />
          <StatusBlock
            v-else-if="searchError"
            type="error"
            title="搜索失败"
            :message="searchError"
            class="compact-state"
          />
          <StatusBlock
            v-else-if="hasSearched && searchResults.length === 0"
            title="未找到匹配商品"
            class="compact-state"
          />
          <div v-else class="article-search-list">
            <button
              v-for="article in searchResults"
              :key="article.article_id"
              type="button"
              class="article-search-row"
              :class="{ active: article.article_id === activeArticleId }"
              @click="selectArticle(article.article_id)"
            >
              <span>{{ article.article_id }}</span>
              <strong>{{ article.prod_name ?? article.product_type_name ?? "--" }}</strong>
              <em>{{ article.colour_group_name ?? "--" }}</em>
            </button>
          </div>
        </Panel>

        <Panel title="选中商品摘要" :subtitle="activeArticleId ?? '尚未选择商品'">
          <div v-if="selectedArticle" class="selected-article">
            <strong>{{ selectedArticle.article.article_id }}</strong>
            <span>{{ selectedArticle.article.prod_name ?? "--" }}</span>
            <span>{{ selectedArticle.article.product_type_name ?? "--" }}</span>
            <span>{{ selectedArticle.article.colour_group_name ?? "--" }}</span>
            <span>{{ selectedArticle.article.garment_group_name ?? "--" }}</span>
          </div>
          <StatusBlock
            v-else
            title="搜索或选择一个商品查看属性图"
            class="compact-state"
          />
        </Panel>
      </aside>

      <section class="graph-main-workspace" aria-label="商品属性图画布">
        <StatusBlock
          v-if="graphLoading"
          type="loading"
          title="加载属性图..."
          class="graph-empty"
        />
        <StatusBlock
          v-else-if="graphError"
          type="error"
          title="属性图加载失败"
          :message="graphError"
          class="graph-empty"
        />
        <ArticleAttributeGraph
          v-else-if="graph"
          :article="graph.article"
          :nodes="graph.nodes"
          :edges="graph.edges"
          :selected-node-id="selectedNodeId"
          :active-type="activeType"
          @select-node="selectNode"
        />
        <StatusBlock
          v-else
          title="搜索或选择一个商品查看属性图"
          message="从左侧搜索结果或属性详情页入口进入"
          class="graph-empty"
        />
      </section>

      <GraphNodeInspector
        :article="graph?.article ?? null"
        :nodes="graph?.nodes ?? []"
        :edges="graph?.edges ?? []"
        :selected-node-id="selectedNodeId"
        :active-type="activeType"
        :source-week="sourceWeek"
        @select-type="selectType"
      />
    </div>

    <GraphEdgeTable
      v-if="graph"
      :nodes="graph.nodes"
      :edges="graph.edges"
      :active-type="activeType"
      :source-week="sourceWeek"
    />
  </section>
</template>

<script setup lang="ts">
import { RotateCcw, Search } from "lucide-vue-next";
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";

import {
  getArticle,
  getArticleGraph,
  getAttributeArticles,
  searchArticles,
} from "@/api/defenseApi";
import ArticleAttributeGraph from "@/components/graph/ArticleAttributeGraph.vue";
import GraphEdgeTable from "@/components/graph/GraphEdgeTable.vue";
import GraphNodeInspector from "@/components/graph/GraphNodeInspector.vue";
import { graphTypeFilters } from "@/components/graph/graphTypes";
import PageToolbar from "@/components/layout/PageToolbar.vue";
import Panel from "@/components/ui/Panel.vue";
import StatusBlock from "@/components/ui/StatusBlock.vue";
import type {
  ArticleDetailResponse,
  ArticleGraphResponse,
  ArticleItem,
} from "@/types/api";

const props = defineProps<{
  articleId?: string;
}>();

const route = useRoute();
const router = useRouter();
const searchTerm = ref("");
const searchResults = ref<ArticleItem[]>([]);
const selectedArticle = ref<ArticleDetailResponse | null>(null);
const graph = ref<ArticleGraphResponse | null>(null);
const selectedNodeId = ref<string | null>(null);
const activeType = ref("all");
const searchLoading = ref(false);
const graphLoading = ref(false);
const representativeLoading = ref(false);
const searchError = ref<string | null>(null);
const graphError = ref<string | null>(null);
const representativeError = ref<string | null>(null);
const hasSearched = ref(false);
let graphRequestId = 0;
let representativeRequestId = 0;

const activeArticleId = computed(() => selectedArticle.value?.article.article_id ?? null);
const queryAttrId = computed(() => readString(route.query.attr_id));
const sourceWeek = computed(() => readOptionalInt(route.query.source_week));
const representativeMessage = computed(() =>
  queryAttrId.value ? `属性 ${queryAttrId.value} 的代表商品` : undefined,
);

const toolbarContext = computed(() => {
  if (!graph.value || !selectedArticle.value) {
    return queryAttrId.value
      ? `属性 ${queryAttrId.value} · 等待代表商品`
      : "搜索或选择一个商品查看属性图";
  }
  return `商品 ${selectedArticle.value.article.article_id} · ${graph.value.nodes.length} 个属性节点 · ${graph.value.edges.length} 条边`;
});

onMounted(() => {
  void syncRouteContext();
});

watch(
  () => [props.articleId, route.query.attr_id],
  () => {
    void syncRouteContext();
  },
);

async function syncRouteContext() {
  representativeError.value = null;
  if (props.articleId) {
    await loadArticleGraph(props.articleId);
    return;
  }
  if (queryAttrId.value) {
    await loadRepresentativeArticle(queryAttrId.value);
    return;
  }
  resetSelectedGraph();
}

async function runSearch() {
  const query = searchTerm.value.trim();
  hasSearched.value = true;
  searchError.value = null;
  searchResults.value = [];
  if (!query) {
    return;
  }

  searchLoading.value = true;
  try {
    searchResults.value = (await searchArticles(query, 10)).items;
  } catch (error) {
    searchError.value = getErrorMessage(error);
  } finally {
    searchLoading.value = false;
  }
}

function selectArticle(articleId: string) {
  const nextQuery = { ...route.query };
  delete nextQuery.attr_id;
  void router.push({
    name: "article-graph",
    params: { articleId },
    query: nextQuery,
  });
}

async function loadRepresentativeArticle(attrId: string) {
  const requestId = ++representativeRequestId;
  representativeLoading.value = true;
  representativeError.value = null;
  try {
    const response = await getAttributeArticles(attrId, 1);
    if (requestId !== representativeRequestId) {
      return;
    }
    const representative = response.items[0];
    if (!representative) {
      resetSelectedGraph();
      representativeError.value = `属性 ${attrId} 暂无可用代表商品`;
      return;
    }
    await router.replace({
      name: "article-graph",
      params: { articleId: representative.article_id },
      query: route.query,
    });
  } catch (error) {
    if (requestId !== representativeRequestId) {
      return;
    }
    representativeError.value = getErrorMessage(error);
  } finally {
    if (requestId === representativeRequestId) {
      representativeLoading.value = false;
    }
  }
}

async function loadArticleGraph(articleId: string) {
  const requestId = ++graphRequestId;
  graphLoading.value = true;
  graphError.value = null;
  selectedArticle.value = null;
  graph.value = null;
  selectedNodeId.value = null;
  try {
    const [articleResponse, graphResponse] = await Promise.all([
      getArticle(articleId),
      getArticleGraph(articleId),
    ]);
    if (requestId !== graphRequestId) {
      return;
    }
    selectedArticle.value = articleResponse;
    graph.value = graphResponse;
    selectedNodeId.value = graphResponse.article.id;
  } catch (error) {
    if (requestId !== graphRequestId) {
      return;
    }
    graphError.value = getErrorMessage(error);
  } finally {
    if (requestId === graphRequestId) {
      graphLoading.value = false;
    }
  }
}

function selectNode(nodeId: string) {
  selectedNodeId.value = nodeId;
}

function selectType(type: string) {
  activeType.value = type || "all";
}

function resetView() {
  activeType.value = "all";
  selectedNodeId.value = graph.value?.article.id ?? null;
}

function resetSelectedGraph() {
  graphRequestId += 1;
  selectedArticle.value = null;
  graph.value = null;
  selectedNodeId.value = null;
  graphError.value = null;
  graphLoading.value = false;
}

function readString(value: unknown) {
  if (typeof value === "string" && value.trim()) {
    return value;
  }
  return null;
}

function readOptionalInt(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function getErrorMessage(value: unknown) {
  return value instanceof Error ? value.message : "请求失败";
}
</script>
