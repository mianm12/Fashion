<template>
  <section class="route-page">
    <header class="page-header">
      <p class="eyebrow">Recommendation</p>
      <h1>推荐案例</h1>
      <div class="search-strip">
        <input
          type="search"
          aria-label="演示用户搜索"
          placeholder="case_id / customer_id"
        />
        <button type="button">Filter</button>
      </div>
    </header>

    <div class="skeleton-grid two-columns wide-left">
      <article class="panel">
        <h2>Demo users</h2>
        <div class="placeholder-list">
          <span v-for="rank in 6" :key="rank">Case {{ rank }}</span>
        </div>
      </article>
      <article class="panel">
        <h2>Top 12</h2>
        <div class="recommend-grid">
          <span v-for="rank in 12" :key="rank">#{{ rank }}</span>
        </div>
      </article>
    </div>
  </section>
</template>
