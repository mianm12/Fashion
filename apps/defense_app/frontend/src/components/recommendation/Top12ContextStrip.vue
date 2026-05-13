<template>
  <div class="top12-context-strip">
    <StatusBlock
      v-if="items.length === 0"
      title="暂无 Top-12 上下文"
      message="当前案例没有可展示的推荐列表"
      class="compact-state"
    />
    <div v-else class="top12-context-table">
      <div class="top12-context-row top12-context-head">
        <span>排名</span>
        <span>商品 ID</span>
        <span>商品名称</span>
        <span>分数</span>
        <span>命中</span>
        <span>操作</span>
      </div>
      <RouterLink
        v-for="item in items"
        :key="`${item.rank}-${item.article_id}`"
        class="top12-context-row"
        :class="{ active: item.article_id === selectedArticleId }"
        :to="{
          name: 'recommendation-explanation',
          params: { caseId, articleId: item.article_id },
          query: returnQuery,
        }"
      >
        <strong>#{{ item.rank }}</strong>
        <span>{{ item.article_id }}</span>
        <span>{{ item.article.prod_name ?? item.article.product_type_name ?? "--" }}</span>
        <span>{{ formatScore(item.score) }}</span>
        <span :class="item.is_hit ? 'hit-marker' : 'miss-marker'">
          {{ item.is_hit ? "命中" : "未命中" }}
        </span>
        <span>切换解释</span>
      </RouterLink>
    </div>
  </div>
</template>

<script setup lang="ts">
import StatusBlock from "@/components/ui/StatusBlock.vue";
import type { RecommendationItem } from "@/types/api";
import type { LocationQueryRaw } from "vue-router";

defineProps<{
  caseId: string;
  items: RecommendationItem[];
  selectedArticleId?: string | null;
  returnQuery?: LocationQueryRaw;
}>();

function formatScore(value: number) {
  return Number.isFinite(value) ? value.toFixed(4) : "--";
}
</script>
