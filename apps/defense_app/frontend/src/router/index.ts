import { createRouter, createWebHistory } from "vue-router";

const TrendDashboardView = () => import("@/views/TrendDashboardView.vue");
const AttributeDetailView = () => import("@/views/AttributeDetailView.vue");
const AttributeGraphView = () => import("@/views/AttributeGraphView.vue");
const RecommendationView = () => import("@/views/RecommendationView.vue");
const UserProfileReasonView = () => import("@/views/UserProfileReasonView.vue");

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/",
      name: "trends",
      component: TrendDashboardView,
    },
    {
      path: "/attributes/:attrId",
      name: "attribute-detail",
      component: AttributeDetailView,
      props: true,
    },
    {
      path: "/graph/articles/:articleId?",
      name: "article-graph",
      component: AttributeGraphView,
      props: true,
    },
    {
      path: "/recommendations",
      name: "recommendations",
      component: RecommendationView,
    },
    {
      path: "/recommendations/:caseId/:articleId?",
      name: "recommendation-explanation",
      component: UserProfileReasonView,
      props: true,
    },
  ],
});

export default router;
