<template>
  <section class="route-page recommendation-page">
    <PageToolbar title="推荐展示" context="预置演示用户池 · pop_similarity_trend · Top-12">
      <template #actions>
        <form class="recommendation-toolbar-search" @submit.prevent="applyFilters">
          <input
            v-model="searchTerm"
            type="search"
            aria-label="演示用户搜索"
            placeholder="案例 ID / 用户 ID"
          />
          <button type="submit" class="toolbar-button">
            <Search aria-hidden="true" />
            筛选
          </button>
        </form>
        <label class="compact-field recommendation-tag-field">
          <span>标签</span>
          <select v-model="tagFilter" @change="applyFilters">
            <option value="">全部</option>
            <option v-for="tag in availableTags" :key="tag" :value="tag">
              {{ tag }}
            </option>
          </select>
        </label>
        <label class="compact-field recommendation-sort-field">
          <span>排序</span>
          <select v-model="sortMode" @change="updateQuery(selectedCaseId)">
            <option value="hits">命中数优先</option>
            <option value="case_id">案例 ID</option>
          </select>
        </label>
        <button type="button" class="toolbar-button" @click="loadUsers">
          <RefreshCw aria-hidden="true" />
          刷新
        </button>
      </template>
    </PageToolbar>

    <div class="recommendation-workspace">
      <Panel title="演示用户池" :subtitle="userPoolSubtitle">
        <template #actions>
          <span class="count-pill">{{ sortedUsers.length }} 个案例</span>
        </template>
        <DemoUserList
          :users="sortedUsers"
          :selected-case-id="selectedCaseId"
          :loading="usersLoading"
          :error="usersError"
          @select="selectUser"
        />
      </Panel>

      <Panel title="Top-12 推荐结果" :subtitle="selectedCaseId ?? '暂无选中用户'">
        <template #actions>
          <span class="count-pill">{{ recommendations.length }} 件商品</span>
        </template>
        <div v-if="selectedUser" class="selected-user-summary">
          <div>
            <span>案例 ID</span>
            <strong>{{ selectedUser.case_id }}</strong>
          </div>
          <div>
            <span>用户 ID</span>
            <strong>{{ shortCustomerId(selectedUser.customer_id) }}</strong>
          </div>
          <div>
            <span>命中数</span>
            <strong>{{ selectedUser.hit_count }}</strong>
          </div>
          <div>
            <span>评估窗口</span>
            <strong>W{{ selectedUser.cutoff_week }} -> W{{ selectedUser.label_week }}</strong>
          </div>
          <p>{{ selectedUser.profile_summary }}</p>
          <p>{{ selectedUser.recommendation_summary }}</p>
        </div>
        <RecommendationGrid
          v-if="selectedCaseId"
          :case-id="selectedCaseId"
          :items="recommendations"
          :loading="recommendationsLoading"
          :error="recommendationsError"
          :return-query="route.query"
        />
        <StatusBlock
          v-else
          title="暂无选中用户"
          message="从左侧演示用户池选择一个案例 ID"
          class="compact-state"
        />
      </Panel>
    </div>

    <div class="recommendation-evidence-grid">
      <Panel title="推荐方法指标" subtitle="稳定推荐方法在 test 切分上的核心指标">
        <RecommendationMetricsStrip
          :metrics="recommendationMetrics"
          :loading="metricsLoading"
          :error="metricsError"
        />
      </Panel>

      <Panel title="推荐明细预览" subtitle="Top-12 前 5 行审计视图">
        <StatusBlock
          v-if="recommendationsLoading"
          type="loading"
          title="加载推荐明细..."
          class="compact-state"
        />
        <StatusBlock
          v-else-if="recommendations.length === 0"
          title="暂无推荐明细"
          class="compact-state"
        />
        <div v-else class="recommendation-preview-table">
          <div class="recommendation-preview-row recommendation-preview-head">
            <span>排名</span>
            <span>商品 ID</span>
            <span>分数</span>
            <span>是否命中</span>
            <span>候选来源</span>
            <span>核心属性</span>
            <span>操作</span>
          </div>
          <div
            v-for="item in recommendationPreviewRows"
            :key="`${item.rank}-${item.article_id}`"
            class="recommendation-preview-row"
          >
            <strong>#{{ item.rank }}</strong>
            <span>{{ item.article_id }}</span>
            <span>{{ formatScore(item.score) }}</span>
            <span :class="item.is_hit ? 'hit-marker' : 'miss-marker'">
              {{ item.is_hit ? "命中" : "未命中" }}
            </span>
            <span>{{ item.candidate_sources }}</span>
            <span>{{ coreAttrs(item) }}</span>
            <RouterLink
              :to="{
                name: 'recommendation-explanation',
                params: { caseId: selectedCaseId, articleId: item.article_id },
                query: route.query,
              }"
            >
              推荐理由
            </RouterLink>
          </div>
        </div>
      </Panel>
    </div>
  </section>
</template>

<script setup lang="ts">
import { RefreshCw, Search } from "lucide-vue-next";
import { computed, onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";

import {
  getRecommendations,
  listDemoUsers,
  listRecommendationMetrics,
} from "@/api/defenseApi";
import PageToolbar from "@/components/layout/PageToolbar.vue";
import DemoUserList from "@/components/recommendation/DemoUserList.vue";
import RecommendationGrid from "@/components/recommendation/RecommendationGrid.vue";
import RecommendationMetricsStrip from "@/components/recommendation/RecommendationMetricsStrip.vue";
import Panel from "@/components/ui/Panel.vue";
import StatusBlock from "@/components/ui/StatusBlock.vue";
import type { DemoUserItem, MetricItem, RecommendationItem } from "@/types/api";

const route = useRoute();
const router = useRouter();

const searchTerm = ref(readQueryString(route.query.q));
const tagFilter = ref(readQueryString(route.query.tag));
const sortMode = ref(readSortMode(route.query.sort));
const users = ref<DemoUserItem[]>([]);
const recommendations = ref<RecommendationItem[]>([]);
const recommendationMetrics = ref<MetricItem[]>([]);
const selectedCaseId = ref(readQueryString(route.query.case_id) || null);
const usersLoading = ref(false);
const recommendationsLoading = ref(false);
const metricsLoading = ref(false);
const usersError = ref<string | null>(null);
const recommendationsError = ref<string | null>(null);
const metricsError = ref<string | null>(null);
let usersRequestId = 0;
let recommendationsRequestId = 0;
let metricsRequestId = 0;

const sortedUsers = computed(() => {
  const items = [...users.value];
  if (sortMode.value === "case_id") {
    return items.sort((left, right) => left.case_id.localeCompare(right.case_id));
  }
  return items.sort(
    (left, right) =>
      right.hit_count - left.hit_count || left.case_id.localeCompare(right.case_id),
  );
});

const selectedUser = computed(
  () => users.value.find((user) => user.case_id === selectedCaseId.value) ?? null,
);

const recommendationPreviewRows = computed(() => recommendations.value.slice(0, 5));

const availableTags = computed(() => {
  const tags = new Set<string>();
  for (const user of users.value) {
    for (const tag of splitTags(user.primary_tags)) {
      tags.add(tag);
    }
  }
  if (tagFilter.value) {
    tags.add(tagFilter.value);
  }
  return [...tags].sort();
});

const userPoolSubtitle = computed(() => {
  const tag = tagFilter.value || "全部";
  const sort = sortMode.value === "case_id" ? "案例 ID" : "命中数优先";
  return `筛选：${tag} · 排序：${sort}`;
});

onMounted(() => {
  void loadUsers();
  void loadMetrics();
});

async function applyFilters() {
  updateQuery(null);
  await loadUsers();
}

async function loadUsers() {
  const requestId = ++usersRequestId;
  usersLoading.value = true;
  usersError.value = null;
  recommendations.value = [];

  try {
    const response = await listDemoUsers({
      q: searchTerm.value || undefined,
      tag: tagFilter.value || undefined,
      limit: 20,
    });
    if (requestId !== usersRequestId) {
      return;
    }
    users.value = response.items;

    const currentExists = response.items.some(
      (user) => user.case_id === selectedCaseId.value,
    );
    selectedCaseId.value = currentExists
      ? selectedCaseId.value
      : response.items[0]?.case_id ?? null;
    updateQuery(selectedCaseId.value);

    if (selectedCaseId.value) {
      await loadRecommendations(selectedCaseId.value);
    }
  } catch (error) {
    if (requestId !== usersRequestId) {
      return;
    }
    usersError.value = getErrorMessage(error);
    users.value = [];
    recommendations.value = [];
    selectedCaseId.value = null;
  } finally {
    if (requestId === usersRequestId) {
      usersLoading.value = false;
    }
  }
}

async function loadMetrics() {
  const requestId = ++metricsRequestId;
  metricsLoading.value = true;
  metricsError.value = null;
  try {
    const response = await listRecommendationMetrics("test");
    if (requestId !== metricsRequestId) {
      return;
    }
    recommendationMetrics.value = response.items;
  } catch (error) {
    if (requestId !== metricsRequestId) {
      return;
    }
    metricsError.value = getErrorMessage(error);
    recommendationMetrics.value = [];
  } finally {
    if (requestId === metricsRequestId) {
      metricsLoading.value = false;
    }
  }
}

async function selectUser(caseId: string) {
  selectedCaseId.value = caseId;
  updateQuery(caseId);
  await loadRecommendations(caseId);
}

async function loadRecommendations(caseId: string) {
  const requestId = ++recommendationsRequestId;
  recommendationsLoading.value = true;
  recommendationsError.value = null;

  try {
    const response = await getRecommendations(caseId);
    if (requestId !== recommendationsRequestId) {
      return;
    }
    recommendations.value = response.items;
  } catch (error) {
    if (requestId !== recommendationsRequestId) {
      return;
    }
    recommendationsError.value = getErrorMessage(error);
    recommendations.value = [];
  } finally {
    if (requestId === recommendationsRequestId) {
      recommendationsLoading.value = false;
    }
  }
}

function updateQuery(caseId: string | null) {
  void router.replace({
    query: {
      ...route.query,
      q: searchTerm.value || undefined,
      tag: tagFilter.value || undefined,
      sort: sortMode.value === "hits" ? undefined : sortMode.value,
      case_id: caseId ?? undefined,
    },
  });
}

function splitTags(value: string) {
  try {
    const parsed = JSON.parse(value) as unknown;
    if (Array.isArray(parsed)) {
      return parsed.filter((item): item is string => typeof item === "string");
    }
  } catch {
    // Backend tests also cover legacy comma-separated tags.
  }
  return value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
}

function coreAttrs(item: RecommendationItem) {
  return [
    item.article.product_type_name,
    item.article.colour_group_name,
    item.article.garment_group_name,
  ]
    .filter(Boolean)
    .join(" / ");
}

function shortCustomerId(customerId: string) {
  return customerId.length > 16
    ? `${customerId.slice(0, 8)}...${customerId.slice(-6)}`
    : customerId;
}

function formatScore(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : "--";
}

function readQueryString(value: unknown) {
  return typeof value === "string" ? value : "";
}

function readSortMode(value: unknown) {
  return value === "case_id" ? "case_id" : "hits";
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "请求失败";
}
</script>
