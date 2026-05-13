<template>
  <div>
    <StatusBlock
      v-if="loading"
      type="loading"
      title="加载 Top-12 推荐..."
      class="compact-state"
    />
    <StatusBlock
      v-else-if="error"
      type="error"
      title="Top-12 推荐加载失败"
      :message="error"
      class="compact-state"
    />
    <StatusBlock
      v-else-if="items.length === 0"
      title="暂无推荐商品"
      class="compact-state"
    />
    <template v-else>
      <div class="recommendation-warning-stack">
        <div v-if="items.length !== 12" class="recommendation-warning">
          Top-12 不完整：当前仅 {{ items.length }} 件商品
        </div>
        <div v-if="duplicateCount > 0" class="recommendation-warning error">
          后端返回 {{ duplicateCount }} 个重复推荐商品，请检查展示库契约
        </div>
      </div>
      <div class="recommendation-card-grid">
        <RouterLink
          v-for="item in items"
          :key="`${item.rank}-${item.article_id}`"
          class="recommendation-card"
          :to="{
            name: 'recommendation-explanation',
            params: { caseId, articleId: item.article_id },
            query: returnQuery,
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
          <div class="candidate-source-row">
            <span v-for="source in candidateSources(item.candidate_sources)" :key="source">
              {{ source }}
            </span>
          </div>
          <footer>
            <span :class="item.is_hit ? 'hit-marker' : 'miss-marker'">
              {{ item.is_hit ? "命中" : "未命中" }}
            </span>
            <span>{{ item.article.product_type_name ?? "--" }}</span>
          </footer>
        </RouterLink>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { LocationQueryRaw } from "vue-router";

import StatusBlock from "@/components/ui/StatusBlock.vue";
import type { RecommendationItem } from "@/types/api";
import { parseJsonStringArray } from "@/utils/formatters";

const props = defineProps<{
  caseId: string;
  items: RecommendationItem[];
  loading?: boolean;
  error?: string | null;
  returnQuery?: LocationQueryRaw;
}>();

const duplicateCount = computed(() => {
  const seen = new Set<string>();
  let count = 0;
  for (const item of props.items) {
    if (seen.has(item.article_id)) {
      count += 1;
      continue;
    }
    seen.add(item.article_id);
  }
  return count;
});

function formatScore(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : "--";
}

function candidateSources(value: string) {
  const parsed = parseJsonStringArray(value);
  return parsed.length ? parsed : [value];
}
</script>
