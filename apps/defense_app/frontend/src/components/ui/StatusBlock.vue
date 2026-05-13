<template>
  <div class="ui-status-block" :class="`ui-status-${type}`" role="status">
    <strong>{{ title }}</strong>
    <span v-if="message">{{ message }}</span>
  </div>
</template>

<script setup lang="ts">
withDefaults(
  defineProps<{
    type?: "loading" | "empty" | "error";
    title: string;
    message?: string;
  }>(),
  {
    type: "empty",
  },
);
</script>
