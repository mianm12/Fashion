export const graphTypeFilters = [
  { value: "all", label: "全部" },
  { value: "colour_group_name", label: "颜色" },
  { value: "product_type_name", label: "品类" },
  { value: "graphical_appearance_name", label: "图案" },
  { value: "garment_group_name", label: "服装组" },
  { value: "department_name", label: "部门" },
] as const;

export type GraphTypeFilter = (typeof graphTypeFilters)[number]["value"];

const graphTypeLabels: Record<string, string> = {
  article: "商品",
  colour_group_name: "颜色",
  product_type_name: "品类",
  graphical_appearance_name: "图案",
  garment_group_name: "服装组",
  department_name: "部门",
  section_name: "Section",
  index_name: "Index",
  index_group_name: "Index Group",
};

export function graphTypeLabel(type: string | null | undefined) {
  if (!type) {
    return "--";
  }
  return graphTypeLabels[type] ?? type.replace(/_/g, " ");
}

export function graphTypeTone(type: string | null | undefined) {
  if (type === "colour_group_name") {
    return "red";
  }
  if (type === "product_type_name") {
    return "green";
  }
  if (type === "graphical_appearance_name") {
    return "gold";
  }
  if (
    type === "garment_group_name" ||
    type === "department_name" ||
    type === "section_name" ||
    type === "index_name" ||
    type === "index_group_name"
  ) {
    return "blue";
  }
  return "neutral";
}
