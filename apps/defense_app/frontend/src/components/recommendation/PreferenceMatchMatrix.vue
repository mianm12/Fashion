<template>
  <div class="preference-match-matrix">
    <StatusBlock
      v-if="matchRows.length === 0"
      title="暂无偏好匹配"
      message="当前商品属性未命中用户画像属性或同类型上下文"
      class="compact-state"
    />
    <div v-else class="preference-match-table">
      <div class="preference-match-row preference-match-head">
        <span>用户偏好</span>
        <span>关系</span>
        <span>商品属性</span>
        <span>状态</span>
      </div>
      <div
        v-for="row in matchRows"
        :key="`${row.profile.attr_id}-${row.item.attr_id}-${row.status}`"
        class="preference-match-row"
      >
        <div>
          <strong>{{ row.profile.attr_value }}</strong>
          <span>{{ row.profile.attr_type }}</span>
        </div>
        <span class="match-arrow">-></span>
        <div>
          <strong>{{ row.item.attr_value }}</strong>
          <span>{{ row.item.attr_type }}</span>
        </div>
        <em :class="row.status === 'strong' ? 'match-strong' : 'match-weak'">
          {{ row.status === "strong" ? "精确匹配" : "同类关联" }}
        </em>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

import StatusBlock from "@/components/ui/StatusBlock.vue";
import type { AttributeItem, UserProfileAttribute } from "@/types/api";

const props = defineProps<{
  profile: UserProfileAttribute[];
  itemAttributes: AttributeItem[];
}>();

const matchRows = computed(() => {
  const rows: Array<{
    profile: UserProfileAttribute;
    item: AttributeItem;
    status: "strong" | "weak";
  }> = [];
  for (const item of props.itemAttributes) {
    const exact = props.profile.find((profile) => profile.attr_id === item.attr_id);
    if (exact) {
      rows.push({ profile: exact, item, status: "strong" });
      continue;
    }
    const sameType = props.profile.find(
      (profile) => profile.attr_type === item.attr_type,
    );
    if (sameType) {
      rows.push({ profile: sameType, item, status: "weak" });
    }
  }
  return rows.slice(0, 10);
});
</script>
