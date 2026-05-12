<template>
  <section class="route-page">
    <header class="page-header">
      <p class="eyebrow">Explanation</p>
      <h1>{{ caseId }}</h1>
      <div class="header-metrics" aria-label="解释占位指标">
        <span v-if="articleId">Article {{ articleId }}</span>
        <span>Final score --</span>
        <span>Hit --</span>
      </div>
    </header>

    <div class="skeleton-grid three-columns">
      <article class="panel">
        <h2>User profile</h2>
        <div class="placeholder-list">
          <span v-for="rank in 5" :key="rank">Preference {{ rank }}</span>
        </div>
      </article>
      <article class="panel">
        <h2>Item attributes</h2>
        <div class="placeholder-list">
          <span v-for="rank in 5" :key="rank">Attribute {{ rank }}</span>
        </div>
      </article>
      <article class="panel">
        <h2>Scores</h2>
        <div class="score-bars">
          <span v-for="rank in 5" :key="rank" />
        </div>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  caseId: string;
  articleId?: string;
}>();
</script>
