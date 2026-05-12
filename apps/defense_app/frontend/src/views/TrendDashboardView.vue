<template>
  <section class="route-page">
    <header class="page-header">
      <p class="eyebrow">Trend</p>
      <h1>趋势看板</h1>
      <div class="header-metrics" aria-label="趋势占位指标">
        <span>Source week --</span>
        <span>Target week --</span>
        <span>Top K --</span>
      </div>
    </header>

    <div class="skeleton-grid four-columns">
      <article
        v-for="attrType in attrTypes"
        :key="attrType"
        class="panel"
      >
        <h2>{{ attrType }}</h2>
        <div class="placeholder-list">
          <span v-for="rank in 5" :key="rank">#{{ rank }}</span>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
const attrTypes = [
  "colour_group_name",
  "product_type_name",
  "graphical_appearance_name",
  "garment_group_name",
];
</script>
