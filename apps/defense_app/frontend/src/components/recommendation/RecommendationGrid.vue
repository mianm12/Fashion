<template>
  <div>
    <div v-if="loading" class="dense-state compact-state">
      加载 Top-12 推荐...
    </div>
    <div v-else-if="error" class="dense-state error-state compact-state">
      {{ error }}
    </div>
    <div v-else-if="items.length === 0" class="dense-state compact-state">
      暂无推荐商品
    </div>
    <template v-else>
      <div class="recommendation-card-grid">
        <div v-if="duplicateCount > 0" class="dense-state error-state compact-state">
          发现 {{ duplicateCount }} 个重复推荐商品，已按 article_id 展示首个
        </div>
        <RouterLink
          v-for="item in uniqueItems"
          :key="item.article_id"
          class="recommendation-card"
          :to="{
            name: 'recommendation-explanation',
            params: { caseId, articleId: item.article_id },
          }"
        >
          <header>
            <span class="rank-cell">#{{ item.rank }}</span>
            <span class="score-chip">{{ formatScore(item.score) }}</span>
          </header>
          <strong>{{ item.article.prod_name ?? item.article_id }}</strong>
          <span class="article-id">{{ item.article_id }}</span>
          <div class="core-attrs">
            <span>{{ item.article.product_type_name ?? "--" }}</span>
            <span>{{ item.article.colour_group_name ?? "--" }}</span>
            <span>{{ item.article.garment_group_name ?? "--" }}</span>
          </div>
          <footer>
            <span :class="item.is_hit ? 'hit-marker' : 'miss-marker'">
              {{ item.is_hit ? "hit" : "not hit" }}
            </span>
            <span>{{ item.candidate_sources }}</span>
          </footer>
        </RouterLink>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

import type { RecommendationItem } from "@/types/api";

const props = defineProps<{
  caseId: string;
  items: RecommendationItem[];
  loading?: boolean;
  error?: string | null;
}>();

const uniqueItems = computed(() => {
  const seen = new Set<string>();
  return props.items.filter((item) => {
    if (seen.has(item.article_id)) {
      return false;
    }
    seen.add(item.article_id);
    return true;
  });
});

const duplicateCount = computed(() => props.items.length - uniqueItems.value.length);

function formatScore(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : "--";
}
</script>
