<template>
  <article class="ui-metric-tile" :class="toneClass">
    <div class="ui-metric-icon" aria-hidden="true">
      <slot name="icon" />
    </div>
    <div class="ui-metric-content">
      <span>{{ label }}</span>
      <strong>{{ value }}</strong>
      <small v-if="hint">{{ hint }}</small>
    </div>
  </article>
</template>

<script setup lang="ts">
import { computed } from "vue";

const props = withDefaults(
  defineProps<{
    label: string;
    value: string;
    hint?: string;
    tone?: "red" | "green" | "gold" | "blue" | "neutral";
  }>(),
  {
    tone: "neutral",
  },
);

const toneClass = computed(() => `ui-metric-${props.tone}`);
</script>
