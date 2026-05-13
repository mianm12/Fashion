<template>
  <div class="demo-user-list">
    <StatusBlock
      v-if="loading"
      type="loading"
      title="加载演示用户..."
      class="compact-state"
    />
    <StatusBlock
      v-else-if="error"
      type="error"
      title="演示用户加载失败"
      :message="error"
      class="compact-state"
    />
    <StatusBlock
      v-else-if="users.length === 0"
      title="未找到匹配演示用户"
      class="compact-state"
    />
    <template v-else>
      <button
        v-for="user in users"
        :key="user.case_id"
        type="button"
        class="demo-user-button"
        :class="{ active: user.case_id === selectedCaseId }"
        @click="emit('select', user.case_id)"
      >
        <span class="case-id">{{ user.case_id }}</span>
        <strong>{{ shortCustomerId(user.customer_id) }}</strong>
        <span class="demo-window">{{ user.split }} · W{{ user.cutoff_week }} -> W{{ user.label_week }}</span>
        <span class="hit-count">{{ user.hit_count }} 命中</span>
        <span class="tag-row">
          <span
            v-for="tag in tags(user.primary_tags).slice(0, 3)"
            :key="tag"
            class="tag-pill"
          >
            {{ tag }}
          </span>
        </span>
      </button>
    </template>
  </div>
</template>

<script setup lang="ts">
import StatusBlock from "@/components/ui/StatusBlock.vue";
import type { DemoUserItem } from "@/types/api";

defineProps<{
  users: DemoUserItem[];
  selectedCaseId: string | null;
  loading?: boolean;
  error?: string | null;
}>();

const emit = defineEmits<{
  select: [caseId: string];
}>();

function tags(value: string) {
  try {
    const parsed = JSON.parse(value) as unknown;
    if (Array.isArray(parsed)) {
      return parsed.filter((item): item is string => typeof item === "string");
    }
  } catch {
    // Older tests seed comma-separated tags; keep that path readable.
  }
  return value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
}

function shortCustomerId(customerId: string) {
  return customerId.length > 16
    ? `${customerId.slice(0, 8)}...${customerId.slice(-6)}`
    : customerId;
}
</script>
