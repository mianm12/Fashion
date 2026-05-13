export type CoreTrendAttrType =
  | "colour_group_name"
  | "product_type_name"
  | "graphical_appearance_name"
  | "garment_group_name";

export type TrendAttrFilter = CoreTrendAttrType | "all";

export const coreTrendAttrTypes: CoreTrendAttrType[] = [
  "colour_group_name",
  "product_type_name",
  "graphical_appearance_name",
  "garment_group_name",
];

export const attrTypeLabels: Record<CoreTrendAttrType, string> = {
  colour_group_name: "颜色",
  product_type_name: "品类",
  graphical_appearance_name: "图案",
  garment_group_name: "服装组",
};
