<template>
  <div class="app-shell">
    <aside class="app-sidebar" aria-label="答辩展示导航">
      <RouterLink class="brand-block" to="/" aria-label="Fashion Trend Lab">
        <Grid3X3 class="brand-icon" aria-hidden="true" />
        <span class="brand-copy">
          <strong>Fashion Trend Lab</strong>
          <span>答辩演示</span>
        </span>
      </RouterLink>

      <nav class="side-nav" aria-label="主导航">
        <RouterLink
          v-for="item in navItems"
          :key="item.to"
          v-slot="{ isActive, navigate, href }"
          :to="item.to"
          custom
        >
          <a
            :href="href"
            class="side-nav-link"
            :class="{ active: isActive }"
            @click="navigate"
          >
            <component :is="item.icon" aria-hidden="true" />
            <span>{{ item.label }}</span>
          </a>
        </RouterLink>
      </nav>

      <section class="sidebar-status" aria-label="展示数据状态">
        <h2>数据源</h2>
        <div class="status-line">
          <Database aria-hidden="true" />
          <span>SQLite 就绪</span>
        </div>
        <div class="status-line">
          <Activity aria-hidden="true" />
          <span>LightGBM 稳定结果</span>
        </div>
        <div class="status-line">
          <LockKeyhole aria-hidden="true" />
          <span>推荐方法已发布</span>
        </div>
      </section>
    </aside>

    <main class="content-frame">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import {
  Activity,
  BarChart3,
  Database,
  GitBranch,
  Grid3X3,
  LockKeyhole,
  MessageSquareQuote,
  Shirt,
  Users,
} from "lucide-vue-next";

const navItems = [
  { to: "/", label: "趋势看板", icon: BarChart3 },
  {
    to: "/attributes/colour_group_name::Black",
    label: "属性详情",
    icon: Shirt,
  },
  { to: "/graph/articles", label: "属性图展示", icon: GitBranch },
  { to: "/recommendations", label: "推荐展示", icon: Users },
  {
    to: "/recommendations/demo_test_102_103_03d54cbe4a45/0706016001",
    label: "推荐理由",
    icon: MessageSquareQuote,
  },
];
</script>
