<template>
  <section class="route-page">
    <header class="page-header">
      <p class="eyebrow">Recommendation</p>
      <h1>轻量 Top-N 推荐实验</h1>
      <form class="search-strip" @submit.prevent="applyFilters">
        <input
          v-model="searchTerm"
          type="search"
          aria-label="演示用户搜索"
          placeholder="case_id / customer_id"
        />
        <input
          v-model="tagFilter"
          type="search"
          aria-label="标签筛选"
          placeholder="tag"
        />
        <button type="submit">Filter</button>
      </form>
    </header>

    <div class="tag-row filter-tags" aria-label="推荐案例标签">
      <button
        type="button"
        class="segment-button"
        :class="{ active: tagFilter === '' }"
        @click="selectTag('')"
      >
        全部
      </button>
      <button
        v-for="tag in availableTags"
        :key="tag"
        type="button"
        class="segment-button"
        :class="{ active: tagFilter === tag }"
        @click="selectTag(tag)"
      >
        {{ tag }}
      </button>
    </div>

    <div class="skeleton-grid wide-left">
      <article class="panel">
        <header class="panel-heading">
          <div>
            <h2>演示用户</h2>
            <p>sorted by hit count</p>
          </div>
          <span class="count-pill">{{ users.length }} cases</span>
        </header>
        <DemoUserList
          :users="sortedUsers"
          :selected-case-id="selectedCaseId"
          :loading="usersLoading"
          :error="usersError"
          @select="selectUser"
        />
      </article>

      <article class="panel">
        <header class="panel-heading">
          <div>
            <h2>Top-12 推荐</h2>
            <p>{{ selectedUser?.case_id ?? "select a demo user" }}</p>
          </div>
          <span class="count-pill">{{ recommendations.length }} items</span>
        </header>
        <div v-if="selectedUser" class="summary-text">
          <strong>{{ selectedUser.profile_summary }}</strong>
          <span>{{ selectedUser.recommendation_summary }}</span>
        </div>
        <RecommendationGrid
          v-if="selectedCaseId"
          :case-id="selectedCaseId"
          :items="recommendations"
          :loading="recommendationsLoading"
          :error="recommendationsError"
        />
        <div v-else class="dense-state compact-state">
          选择演示用户查看离线推荐结果
        </div>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";

import { getRecommendations, listDemoUsers } from "@/api/defenseApi";
import DemoUserList from "@/components/recommendation/DemoUserList.vue";
import RecommendationGrid from "@/components/recommendation/RecommendationGrid.vue";
import type { DemoUserItem, RecommendationItem } from "@/types/api";

const route = useRoute();
const router = useRouter();

const searchTerm = ref(readQueryString(route.query.q));
const tagFilter = ref(readQueryString(route.query.tag));
const users = ref<DemoUserItem[]>([]);
const recommendations = ref<RecommendationItem[]>([]);
const selectedCaseId = ref(readQueryString(route.query.case_id) || null);
const usersLoading = ref(false);
const recommendationsLoading = ref(false);
const usersError = ref<string | null>(null);
const recommendationsError = ref<string | null>(null);
let usersRequestId = 0;
let recommendationsRequestId = 0;

const sortedUsers = computed(() =>
  [...users.value].sort(
    (left, right) => right.hit_count - left.hit_count || left.case_id.localeCompare(right.case_id),
  ),
);

const selectedUser = computed(
  () => users.value.find((user) => user.case_id === selectedCaseId.value) ?? null,
);

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

onMounted(() => {
  void loadUsers();
});

async function applyFilters() {
  updateQuery(null);
  await loadUsers();
}

async function selectTag(tag: string) {
  tagFilter.value = tag;
  await applyFilters();
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
    selectedCaseId.value = currentExists ? selectedCaseId.value : response.items[0]?.case_id ?? null;
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

function readQueryString(value: unknown) {
  return typeof value === "string" ? value : "";
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "请求失败";
}
</script>
