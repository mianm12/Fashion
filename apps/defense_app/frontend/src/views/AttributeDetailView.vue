<template>
  <section class="route-page">
    <header class="page-header">
      <p class="eyebrow">Attribute</p>
      <h1>{{ attrId }}</h1>
      <div class="header-metrics" aria-label="属性占位指标">
        <span>Heat --</span>
        <span>Growth --</span>
        <span>Articles --</span>
      </div>
    </header>

    <div class="skeleton-grid two-columns">
      <article class="panel chart-panel">
        <h2>Heat series</h2>
        <div class="chart-placeholder">
          <span v-for="point in 8" :key="point" />
        </div>
      </article>
      <article class="panel">
        <h2>Related articles</h2>
        <div class="placeholder-list">
          <span v-for="rank in 6" :key="rank">Article {{ rank }}</span>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  attrId: string;
}>();
</script>
