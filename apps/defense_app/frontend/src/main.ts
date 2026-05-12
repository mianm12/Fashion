import { createApp } from "vue";

import AppShell from "@/components/layout/AppShell.vue";
import router from "@/router";
import "@/styles/main.css";

createApp(AppShell).use(router).mount("#app");
