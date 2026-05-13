<template>
  <section class="route-page recommendation-reason-page">
    <PageToolbar title="推荐理由" :context="toolbarContext">
      <template #actions>
        <RouterLink
          class="toolbar-button"
          :to="{ name: 'recommendations', query: returnQuery }"
        >
          <ArrowLeft aria-hidden="true" />
          返回推荐展示
        </RouterLink>
        <label class="compact-field top12-switch-field">
          <span>Top-12</span>
          <select v-model="selectedArticleInput" @change="switchArticle">
            <option value="">选择商品</option>
            <option
              v-for="item in recommendations"
              :key="item.article_id"
              :value="item.article_id"
            >
              #{{ item.rank }} · {{ item.article_id }}
            </option>
          </select>
        </label>
        <button type="button" class="toolbar-button" @click="loadExplanationFlow">
          <RefreshCw aria-hidden="true" />
          刷新
        </button>
      </template>
    </PageToolbar>

    <StatusBlock
      v-if="loading"
      type="loading"
      title="加载推荐解释..."
      class="compact-state"
    />
    <StatusBlock
      v-else-if="error"
      type="error"
      title="推荐解释加载失败"
      :message="error"
      class="compact-state"
    />

    <template v-else>
      <div v-if="demoUser" class="reason-summary-strip">
        <strong>{{ demoUser.profile_summary }}</strong>
        <span>{{ demoUser.recommendation_summary }}</span>
      </div>

      <div class="reason-workspace">
        <Panel title="用户画像属性" subtitle="偏好分、购买次数和最近购买周">
          <template #actions>
            <span class="count-pill">{{ profile.length }} 个属性</span>
          </template>
          <StatusBlock
            v-if="profile.length === 0"
            title="暂无用户画像属性"
            class="compact-state"
          />
          <div v-else class="profile-list">
            <div
              v-for="item in sortedProfile"
              :key="item.attr_id"
              class="profile-row reason-profile-row"
              :class="{ matched: matchedProfileIds.has(item.attr_id) }"
            >
              <strong>{{ item.attr_value }}</strong>
              <span>{{ item.attr_type }}</span>
              <span>{{ formatScore(item.preference_score) }}</span>
              <span>{{ item.purchase_count }} 次购买 · W{{ item.last_purchase_week }}</span>
            </div>
          </div>
        </Panel>

        <Panel title="推荐商品属性" :subtitle="articlePanelSubtitle">
          <template #actions>
            <span
              v-if="selectedRecommendation"
              :class="selectedRecommendation.is_hit ? 'hit-marker' : 'miss-marker'"
            >
              {{ selectedRecommendation.is_hit ? "命中" : "未命中" }}
            </span>
          </template>

          <StatusBlock
            v-if="!props.articleId"
            title="请选择 Top-12 商品"
            message="使用顶部切换器或底部上下文选择一个商品"
            class="compact-state"
          />
          <StatusBlock
            v-else-if="!explanation"
            title="暂无商品解释"
            class="compact-state"
          />
          <div v-else class="reason-article-panel">
            <div class="reason-article-summary">
              <strong>{{ explanation.article.article_id }}</strong>
              <span>{{ explanation.article.prod_name ?? "--" }}</span>
              <span>{{ explanation.article.product_type_name ?? "--" }}</span>
              <span>{{ explanation.article.colour_group_name ?? "--" }}</span>
              <span>{{ explanation.article.graphical_appearance_name ?? "--" }}</span>
              <span>{{ explanation.article.garment_group_name ?? "--" }}</span>
              <span>{{ explanation.article.department_name ?? "--" }}</span>
              <span>{{ explanation.article.section_name ?? "--" }}</span>
            </div>
            <div class="reason-item-attribute-list">
              <RouterLink
                v-for="attribute in explanation.item_attributes"
                :key="attribute.attr_id"
                :class="{
                  matched:
                    matchedItemAttrIds.has(attribute.attr_id) ||
                    trendAttrIds.has(attribute.attr_id),
                }"
                :to="{ name: 'attribute-detail', params: { attrId: attribute.attr_id } }"
              >
                <strong>{{ attribute.attr_value }}</strong>
                <span>{{ attribute.attr_type }}</span>
              </RouterLink>
            </div>
            <RouterLink
              class="panel-action-link"
              :to="{
                name: 'article-graph',
                params: { articleId: explanation.article.article_id },
              }"
            >
              查看属性图
            </RouterLink>
          </div>
        </Panel>

        <Panel title="分数分解" subtitle="已有推荐结果的 score components">
          <template #actions>
            <span class="count-pill">#{{ selectedRecommendation?.rank ?? "--" }}</span>
          </template>
          <ScoreBreakdown :scores="explanation?.score_components ?? null" />
        </Panel>
      </div>

      <div class="reason-evidence-grid">
        <Panel title="偏好 × 商品属性匹配" subtitle="仅展示真实 attr_id 或 attr_type 匹配">
          <PreferenceMatchMatrix
            :profile="profile"
            :item-attributes="explanation?.item_attributes ?? []"
          />
        </Panel>

        <Panel title="匹配趋势属性" subtitle="来自 explanation payload 的趋势信号">
          <template #actions>
            <span class="count-pill">
              {{ explanation?.matching_trend_attributes.length ?? 0 }} 个趋势
            </span>
          </template>
          <StatusBlock
            v-if="!explanation || explanation.matching_trend_attributes.length === 0"
            title="暂无匹配趋势属性"
            class="compact-state"
          />
          <div v-else class="reason-trend-table">
            <RouterLink
              v-for="trend in explanation.matching_trend_attributes"
              :key="`${trend.source_week}-${trend.attr_id}`"
              :to="{
                name: 'attribute-detail',
                params: { attrId: trend.attr_id },
                query: { source_week: trend.source_week },
              }"
            >
              <span>#{{ trend.rank }}</span>
              <strong>{{ trend.attr_value }}</strong>
              <span>{{ trend.attr_type }}</span>
              <span>{{ formatSignedPercent(trend.pred_target_growth) }}</span>
              <em>{{ trend.source_week }} -> {{ trend.target_week }}</em>
            </RouterLink>
          </div>
        </Panel>
      </div>

      <Panel title="Top-12 推荐上下文" subtitle="点击其他行切换当前解释商品">
        <template #actions>
          <span class="count-pill">{{ recommendations.length }} 件商品</span>
        </template>
        <Top12ContextStrip
          :case-id="caseId"
          :items="recommendations"
          :selected-article-id="props.articleId ?? null"
          :return-query="route.query"
        />
      </Panel>
    </template>
  </section>
</template>

<script setup lang="ts">
import { ArrowLeft, RefreshCw } from "lucide-vue-next";
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";

import {
  getDemoUser,
  getRecommendationExplanation,
  getRecommendations,
  getUserProfile,
} from "@/api/defenseApi";
import PageToolbar from "@/components/layout/PageToolbar.vue";
import PreferenceMatchMatrix from "@/components/recommendation/PreferenceMatchMatrix.vue";
import ScoreBreakdown from "@/components/recommendation/ScoreBreakdown.vue";
import Top12ContextStrip from "@/components/recommendation/Top12ContextStrip.vue";
import Panel from "@/components/ui/Panel.vue";
import StatusBlock from "@/components/ui/StatusBlock.vue";
import type {
  DemoUserItem,
  RecommendationExplanationResponse,
  RecommendationItem,
  UserProfileAttribute,
} from "@/types/api";
import { formatSignedPercent } from "@/utils/formatters";

const props = defineProps<{
  caseId: string;
  articleId?: string;
}>();

const router = useRouter();
const route = useRoute();
const demoUser = ref<DemoUserItem | null>(null);
const profile = ref<UserProfileAttribute[]>([]);
const recommendations = ref<RecommendationItem[]>([]);
const explanation = ref<RecommendationExplanationResponse | null>(null);
const selectedArticleInput = ref(props.articleId ?? "");
const loading = ref(false);
const error = ref<string | null>(null);
let reasonRequestId = 0;

const selectedRecommendation = computed(
  () =>
    recommendations.value.find((item) => item.article_id === props.articleId) ??
    null,
);

const sortedProfile = computed(() =>
  [...profile.value].sort(
    (left, right) =>
      right.preference_score - left.preference_score ||
      left.attr_value.localeCompare(right.attr_value),
  ),
);

const matchedProfileIds = computed(
  () =>
    new Set(
      profile.value
        .filter((profileItem) =>
          (explanation.value?.item_attributes ?? []).some(
            (attribute) => attribute.attr_id === profileItem.attr_id,
          ),
        )
        .map((item) => item.attr_id),
    ),
);

const matchedItemAttrIds = computed(
  () =>
    new Set(
      (explanation.value?.item_attributes ?? [])
        .filter((attribute) =>
          profile.value.some((profileItem) => profileItem.attr_id === attribute.attr_id),
        )
        .map((attribute) => attribute.attr_id),
    ),
);

const trendAttrIds = computed(
  () =>
    new Set(
      (explanation.value?.matching_trend_attributes ?? []).map(
        (attribute) => attribute.attr_id,
      ),
    ),
);

const toolbarContext = computed(() => {
  const rank = selectedRecommendation.value?.rank
    ? `排名 #${selectedRecommendation.value.rank}`
    : "排名 --";
  const hit = selectedRecommendation.value
    ? selectedRecommendation.value.is_hit
      ? "命中"
      : "未命中"
    : "--";
  return `${props.caseId} · ${props.articleId ?? "未选择商品"} · ${rank} · ${hit}`;
});

const articlePanelSubtitle = computed(() =>
  props.articleId ? `商品 ${props.articleId}` : "请选择推荐商品",
);

const returnQuery = computed(() => ({
  ...route.query,
  case_id: props.caseId,
}));

onMounted(() => {
  void loadExplanationFlow();
});

watch(
  () => [props.caseId, props.articleId],
  () => {
    selectedArticleInput.value = props.articleId ?? "";
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

function switchArticle() {
  if (!selectedArticleInput.value) {
    return;
  }
  void router.push({
    name: "recommendation-explanation",
    params: {
      caseId: props.caseId,
      articleId: selectedArticleInput.value,
    },
    query: route.query,
  });
}

function formatScore(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "--";
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "请求失败";
}
</script>
