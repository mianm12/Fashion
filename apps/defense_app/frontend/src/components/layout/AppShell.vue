<template>
  <div class="app-shell">
    <header class="topbar">
      <RouterLink class="brand" to="/" aria-label="Fashion defense demo">
        <Grid3X3 class="brand-icon" aria-hidden="true" />
        <span>Fashion Defense</span>
      </RouterLink>

      <nav class="topnav" aria-label="主导航">
        <RouterLink
          v-for="item in navItems"
          :key="item.to"
          v-slot="{ isActive, navigate, href }"
          :to="item.to"
          custom
        >
          <a
            :href="href"
            class="nav-link"
            :class="{ active: isActive }"
            @click="navigate"
          >
            <component :is="item.icon" aria-hidden="true" />
            <span>{{ item.label }}</span>
          </a>
        </RouterLink>
      </nav>
    </header>

    <main class="content-frame">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { BarChart3, GitBranch, Grid3X3, Shirt, Users } from "lucide-vue-next";

const navItems = [
  { to: "/", label: "Trends", icon: BarChart3 },
  { to: "/graph/articles", label: "Graph", icon: GitBranch },
  { to: "/recommendations", label: "Recommendations", icon: Users },
  {
    to: "/attributes/colour_group_name::Black",
    label: "Attribute",
    icon: Shirt,
  },
];
</script>
