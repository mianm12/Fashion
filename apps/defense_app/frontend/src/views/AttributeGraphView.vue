<template>
  <section class="route-page">
    <header class="page-header">
      <p class="eyebrow">Graph</p>
      <h1>{{ articleId ? `商品 ${articleId}` : "商品属性图" }}</h1>
      <div class="search-strip">
        <input
          type="search"
          aria-label="商品搜索"
          placeholder="article_id / prod_name"
        />
        <button type="button">Search</button>
      </div>
    </header>

    <div class="graph-skeleton">
      <span class="graph-node center-node">Article</span>
      <span class="graph-node node-a">Colour</span>
      <span class="graph-node node-b">Product</span>
      <span class="graph-node node-c">Garment</span>
      <span class="graph-node node-d">Pattern</span>
    </div>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  articleId?: string;
}>();
</script>
