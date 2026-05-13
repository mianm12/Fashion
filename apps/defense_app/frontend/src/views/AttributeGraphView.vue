<template>
  <section class="route-page">
    <header class="page-header">
      <p class="eyebrow">Graph</p>
      <h1>{{ selectedArticle?.article.prod_name ?? "商品属性图" }}</h1>
      <form class="search-strip" @submit.prevent="runSearch">
        <input
          v-model="searchTerm"
          type="search"
          aria-label="商品搜索"
          placeholder="article_id / prod_name"
        />
        <button type="submit">Search</button>
      </form>
    </header>

    <div class="skeleton-grid wide-left">
      <article class="panel">
        <header class="panel-heading">
          <div>
            <h2>商品搜索</h2>
            <p>select an article to inspect</p>
          </div>
          <span class="count-pill">{{ searchResults.length }} results</span>
        </header>

        <div v-if="searchLoading" class="dense-state compact-state">
          搜索商品...
        </div>
        <div v-else-if="searchError" class="dense-state error-state compact-state">
          {{ searchError }}
        </div>
        <div
          v-else-if="hasSearched && searchResults.length === 0"
          class="dense-state compact-state"
        >
          未找到匹配商品
        </div>
        <div v-else class="article-table compact-list">
          <button
            v-for="article in searchResults"
            :key="article.article_id"
            type="button"
            class="article-row article-button"
            :class="{ active: article.article_id === activeArticleId }"
            @click="selectArticle(article.article_id)"
          >
            <span>{{ article.article_id }}</span>
            <strong>{{ article.prod_name ?? article.product_type_name ?? "--" }}</strong>
            <span>{{ article.colour_group_name ?? "--" }}</span>
          </button>
        </div>
      </article>

      <article class="panel">
        <header class="panel-heading">
          <div>
            <h2>选中商品</h2>
            <p>{{ activeArticleId ?? "no article selected" }}</p>
          </div>
          <span class="count-pill">{{ graph?.nodes.length ?? 0 }} nodes</span>
        </header>
        <div v-if="selectedArticle" class="selected-article">
          <strong>{{ selectedArticle.article.article_id }}</strong>
          <span>{{ selectedArticle.article.prod_name ?? "--" }}</span>
          <span>{{ selectedArticle.article.product_type_name ?? "--" }}</span>
          <span>{{ selectedArticle.article.colour_group_name ?? "--" }}</span>
        </div>
        <div v-else class="dense-state compact-state">
          从左侧搜索结果选择商品
        </div>
      </article>
    </div>

    <div v-if="graphLoading" class="dense-state">加载属性图...</div>
    <div v-else-if="graphError" class="dense-state error-state">{{ graphError }}</div>
    <ArticleAttributeGraph
      v-else-if="graph"
      :article="graph.article"
      :nodes="graph.nodes"
      :edges="graph.edges"
    />
    <div v-else class="dense-state">暂无商品属性图</div>

    <article v-if="graph" class="panel">
      <header class="panel-heading">
        <div>
          <h2>属性分组</h2>
          <p>attribute groups and edge labels</p>
        </div>
        <span class="count-pill">{{ graph.edges.length }} edges</span>
      </header>
      <div class="group-summary">
        <span v-for="group in groupedNodes" :key="group.type">
          {{ group.type }} · {{ group.count }}
        </span>
      </div>
      <div v-if="graph.edges.length === 0" class="dense-state compact-state">
        暂无商品属性边
      </div>
      <ul v-else class="relation-list">
        <li v-for="edge in graph.edges" :key="`${edge.source}-${edge.target}`">
          <span>{{ edge.source }}</span>
          <strong>{{ edge.relation_type }}</strong>
          <span>{{ edge.target }}</span>
        </li>
      </ul>
    </article>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";

import {
  getArticle,
  getArticleGraph,
  searchArticles,
} from "@/api/defenseApi";
import ArticleAttributeGraph from "@/components/graph/ArticleAttributeGraph.vue";
import type {
  ArticleDetailResponse,
  ArticleGraphResponse,
  ArticleItem,
} from "@/types/api";

const props = defineProps<{
  articleId?: string;
}>();

const router = useRouter();
const searchTerm = ref("");
const searchResults = ref<ArticleItem[]>([]);
const selectedArticle = ref<ArticleDetailResponse | null>(null);
const graph = ref<ArticleGraphResponse | null>(null);
const searchLoading = ref(false);
const graphLoading = ref(false);
const searchError = ref<string | null>(null);
const graphError = ref<string | null>(null);
const hasSearched = ref(false);
let graphRequestId = 0;

const activeArticleId = computed(() => selectedArticle.value?.article.article_id ?? null);

const groupedNodes = computed(() => {
  const counts = new Map<string, number>();
  for (const node of graph.value?.nodes ?? []) {
    counts.set(node.type, (counts.get(node.type) ?? 0) + 1);
  }
  return [...counts.entries()].map(([type, count]) => ({ type, count }));
});

onMounted(() => {
  if (props.articleId) {
    void loadArticleGraph(props.articleId);
  }
});

watch(
  () => props.articleId,
  (articleId) => {
    if (articleId) {
      void loadArticleGraph(articleId);
      return;
    }
    resetSelectedGraph();
  },
);

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
  void router.push({ name: "article-graph", params: { articleId } });
}

async function loadArticleGraph(articleId: string) {
  const requestId = ++graphRequestId;
  graphLoading.value = true;
  graphError.value = null;
  selectedArticle.value = null;
  graph.value = null;
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

function resetSelectedGraph() {
  graphRequestId += 1;
  selectedArticle.value = null;
  graph.value = null;
  graphError.value = null;
  graphLoading.value = false;
}

function getErrorMessage(value: unknown) {
  return value instanceof Error ? value.message : "请求失败";
}
</script>
