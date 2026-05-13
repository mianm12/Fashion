<template>
  <header class="page-toolbar">
    <div class="page-toolbar-main">
      <h1>{{ title }}</h1>
      <p v-if="context">{{ context }}</p>
    </div>

    <div class="page-toolbar-actions">
      <slot name="actions" />
      <span v-if="status" class="toolbar-status">{{ status }}</span>
    </div>
  </header>
</template>

<script setup lang="ts">
defineProps<{
  title: string;
  context?: string;
  status?: string;
}>();
</script>
