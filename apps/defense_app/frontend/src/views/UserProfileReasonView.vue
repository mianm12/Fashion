<template>
  <section class="route-page">
    <header class="page-header">
      <p class="eyebrow">Explanation</p>
      <h1>{{ demoUser?.case_id ?? caseId }}</h1>
      <div class="header-metrics">
        <span>{{ demoUser?.split ?? "split" }}</span>
        <span>hits {{ demoUser?.hit_count ?? "--" }}</span>
        <span v-if="articleId">Article {{ articleId }}</span>
        <span>Final {{ formatScore(explanation?.score_components.final_score) }}</span>
      </div>
    </header>

    <div v-if="loading" class="dense-state">加载推荐解释...</div>
    <div v-else-if="error" class="dense-state error-state">{{ error }}</div>

    <template v-else>
      <div v-if="demoUser" class="summary-text explanation-summary">
        <strong>{{ demoUser.profile_summary }}</strong>
        <span>{{ demoUser.recommendation_summary }}</span>
      </div>

      <div class="skeleton-grid three-columns">
        <article class="panel">
          <header class="panel-heading">
            <div>
              <h2>用户画像属性</h2>
              <p>preference score / purchases / last week</p>
            </div>
            <span class="count-pill">{{ profile.length }} attrs</span>
          </header>
          <div v-if="profile.length === 0" class="dense-state compact-state">
            暂无用户画像属性
          </div>
          <div v-else class="profile-list">
            <div v-for="item in profile" :key="item.attr_id" class="profile-row">
              <strong>{{ item.attr_value }}</strong>
              <span>{{ item.attr_type }}</span>
              <span>{{ formatScore(item.preference_score) }}</span>
              <span>{{ item.purchase_count }} buys · W{{ item.last_purchase_week }}</span>
            </div>
          </div>
        </article>

        <article class="panel">
          <header class="panel-heading">
            <div>
              <h2>推荐商品</h2>
              <p>{{ selectedRecommendation?.article_id ?? "select an item" }}</p>
            </div>
            <span
              v-if="selectedRecommendation"
              :class="selectedRecommendation.is_hit ? 'hit-marker' : 'miss-marker'"
            >
              {{ selectedRecommendation.is_hit ? "hit" : "not hit" }}
            </span>
          </header>

          <div v-if="explanation" class="selected-article">
            <strong>{{ explanation.article.article_id }}</strong>
            <span>{{ explanation.article.prod_name ?? "--" }}</span>
            <span>{{ explanation.article.product_type_name ?? "--" }}</span>
            <span>{{ explanation.article.colour_group_name ?? "--" }}</span>
          </div>
          <div v-else class="dense-state compact-state">
            从 Top-12 推荐中选择商品查看 explanation
          </div>
        </article>

        <article class="panel">
          <header class="panel-heading">
            <div>
              <h2>Score breakdown</h2>
              <p>pop / sim / trend / recent / final</p>
            </div>
            <span class="count-pill">{{ selectedRecommendation?.rank ?? "--" }}</span>
          </header>
          <ScoreBreakdown :scores="explanation?.score_components ?? null" />
        </article>
      </div>

      <div class="skeleton-grid two-columns">
        <article class="panel">
          <header class="panel-heading">
            <div>
              <h2>商品属性</h2>
              <p>product attributes used for explanation</p>
            </div>
            <span class="count-pill">{{ explanation?.item_attributes.length ?? 0 }} attrs</span>
          </header>
          <div
            v-if="!explanation || explanation.item_attributes.length === 0"
            class="dense-state compact-state"
          >
            暂无商品属性
          </div>
          <div v-else class="article-attribute-list">
            <RouterLink
              v-for="attribute in explanation.item_attributes"
              :key="attribute.attr_id"
              :to="{ name: 'attribute-detail', params: { attrId: attribute.attr_id } }"
            >
              <strong>{{ attribute.attr_value }}</strong>
              <span>{{ attribute.attr_type }}</span>
            </RouterLink>
          </div>
        </article>

        <article class="panel">
          <header class="panel-heading">
            <div>
              <h2>匹配趋势属性</h2>
              <p>offline trend prediction signals</p>
            </div>
            <span class="count-pill">
              {{ explanation?.matching_trend_attributes.length ?? 0 }} trends
            </span>
          </header>
          <div
            v-if="!explanation || explanation.matching_trend_attributes.length === 0"
            class="dense-state compact-state"
          >
            暂无匹配趋势属性
          </div>
          <ol v-else class="trend-match-list">
            <li
              v-for="trend in explanation.matching_trend_attributes"
              :key="`${trend.source_week}-${trend.attr_id}`"
            >
              <span>#{{ trend.rank }}</span>
              <strong>{{ trend.attr_value }}</strong>
              <span>{{ trend.attr_type }}</span>
              <span>{{ formatPercent(trend.pred_target_growth) }}</span>
            </li>
          </ol>
        </article>
      </div>

      <article class="panel">
        <header class="panel-heading">
          <div>
            <h2>Top-12 推荐候选</h2>
            <p>轻量离线推荐实验结果</p>
          </div>
          <span class="count-pill">{{ recommendations.length }} items</span>
        </header>
        <RecommendationGrid
          :case-id="caseId"
          :items="recommendations"
          :loading="false"
          :error="null"
        />
      </article>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";

import {
  getDemoUser,
  getRecommendationExplanation,
  getRecommendations,
  getUserProfile,
} from "@/api/defenseApi";
import RecommendationGrid from "@/components/recommendation/RecommendationGrid.vue";
import ScoreBreakdown from "@/components/recommendation/ScoreBreakdown.vue";
import type {
  DemoUserItem,
  RecommendationExplanationResponse,
  RecommendationItem,
  UserProfileAttribute,
} from "@/types/api";

const props = defineProps<{
  caseId: string;
  articleId?: string;
}>();

const demoUser = ref<DemoUserItem | null>(null);
const profile = ref<UserProfileAttribute[]>([]);
const recommendations = ref<RecommendationItem[]>([]);
const explanation = ref<RecommendationExplanationResponse | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);
let reasonRequestId = 0;

const selectedRecommendation = computed(
  () =>
    recommendations.value.find((item) => item.article_id === props.articleId) ??
    null,
);

onMounted(() => {
  void loadExplanationFlow();
});

watch(
  () => [props.caseId, props.articleId],
  () => {
    void loadExplanationFlow();
  },
);

async function loadExplanationFlow() {
  const requestId = ++reasonRequestId;
  loading.value = true;
  error.value = null;
  demoUser.value = null;
  profile.value = [];
  recommendations.value = [];
  explanation.value = null;

  try {
    const [userResponse, profileResponse, recommendationsResponse] = await Promise.all([
      getDemoUser(props.caseId),
      getUserProfile(props.caseId),
      getRecommendations(props.caseId),
    ]);
    if (requestId !== reasonRequestId) {
      return;
    }

    demoUser.value = userResponse;
    profile.value = profileResponse.items;
    recommendations.value = recommendationsResponse.items;

    if (props.articleId) {
      const explanationResponse = await getRecommendationExplanation(
        props.caseId,
        props.articleId,
      );
      if (requestId !== reasonRequestId) {
        return;
      }
      explanation.value = explanationResponse;
      profile.value = explanationResponse.user_profile;
    }
  } catch (loadError) {
    if (requestId !== reasonRequestId) {
      return;
    }
    error.value = getErrorMessage(loadError);
  } finally {
    if (requestId === reasonRequestId) {
      loading.value = false;
    }
  }
}

function formatScore(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "--";
}

function formatPercent(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(1)}%`
    : "--";
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "请求失败";
}
</script>
