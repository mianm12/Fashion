from __future__ import annotations

import pandas as pd

from fashion_trend.recommendation.retrieval.popularity import SOURCE_COLUMNS


def build_product_variant_candidates(
    reorder_candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    article_product_map: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    seed_top_n: int = 6,
    per_seed_top_n: int = 3,
    top_n: int = 12,
) -> pd.DataFrame:
    """Return same-product variants seeded from top reorder candidates."""
    if reorder_candidates.empty or article_product_map.empty:
        return _empty_source_frame()

    reorder_candidates = _with_string_ids(reorder_candidates)
    transactions = _with_string_ids(transactions)
    product_map = _clean_product_map(article_product_map)
    if product_map.empty:
        return _empty_source_frame()

    frames: list[pd.DataFrame] = []
    for window in windows.to_dict("records"):
        seeds = _seeds_for_window(
            reorder_candidates,
            product_map,
            window,
            seed_top_n,
        )
        if seeds.empty:
            continue
        variants = _variants_for_seeds(seeds, product_map)
        if variants.empty:
            continue
        popularity = _article_popularity_for_window(
            transactions,
            cutoff_week=int(window["cutoff_week"]),
        )
        limited = _limit_variants_per_seed(variants, popularity, per_seed_top_n)
        ranked = _rank_window_variants(limited, top_n)
        if ranked.empty:
            continue
        ranked.insert(0, "label_week", window["label_week"])
        ranked.insert(0, "cutoff_week", window["cutoff_week"])
        ranked.insert(0, "split", window["split"])
        frames.append(ranked.loc[:, SOURCE_COLUMNS])
    return _concat_source_frames(frames)


def _clean_product_map(article_product_map: pd.DataFrame) -> pd.DataFrame:
    product_map = article_product_map.loc[:, ["article_id", "product_code"]].copy()
    product_map = product_map.loc[product_map["product_code"].notna()].copy()
    product_map["article_id"] = product_map["article_id"].astype(str)
    product_map["product_code"] = product_map["product_code"].astype(str)
    product_map = product_map.loc[product_map["product_code"] != ""]
    return product_map.drop_duplicates().reset_index(drop=True)


def _seeds_for_window(
    reorder_candidates: pd.DataFrame,
    product_map: pd.DataFrame,
    window: dict[str, object],
    seed_top_n: int,
) -> pd.DataFrame:
    source_rank = pd.to_numeric(reorder_candidates["source_rank"], errors="raise")
    mask = (
        (reorder_candidates["split"] == window["split"])
        & (reorder_candidates["cutoff_week"] == window["cutoff_week"])
        & (reorder_candidates["label_week"] == window["label_week"])
        & (reorder_candidates["source"] == "reorder")
        & (source_rank <= seed_top_n)
    )
    seeds = reorder_candidates.loc[
        mask,
        ["customer_id", "article_id", "source_rank"],
    ].copy()
    if seeds.empty:
        return seeds
    seeds["seed_rank"] = pd.to_numeric(seeds["source_rank"], errors="raise").astype(int)
    seeds = seeds.rename(columns={"article_id": "seed_article_id"})
    seeds = seeds.drop(columns=["source_rank"]).drop_duplicates()
    return seeds.merge(
        product_map.rename(columns={"article_id": "seed_article_id"}),
        on="seed_article_id",
        how="inner",
    )


def _variants_for_seeds(
    seeds: pd.DataFrame,
    product_map: pd.DataFrame,
) -> pd.DataFrame:
    variants = seeds.merge(
        product_map.rename(columns={"article_id": "article_id"}),
        on="product_code",
        how="inner",
    )
    return variants.loc[variants["article_id"] != variants["seed_article_id"]].copy()


def _article_popularity_for_window(
    transactions: pd.DataFrame,
    cutoff_week: int,
) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(columns=["article_id", "article_popularity"])
    week_id = pd.to_numeric(transactions["week_id"], errors="raise")
    history = transactions.loc[week_id <= cutoff_week].copy()
    if history.empty:
        return pd.DataFrame(columns=["article_id", "article_popularity"])
    return (
        history.groupby("article_id", as_index=False)
        .size()
        .rename(columns={"size": "article_popularity"})
    )


def _limit_variants_per_seed(
    variants: pd.DataFrame,
    popularity: pd.DataFrame,
    per_seed_top_n: int,
) -> pd.DataFrame:
    scored = _attach_popularity(variants, popularity)
    sorted_variants = scored.sort_values(
        [
            "customer_id",
            "seed_article_id",
            "article_popularity",
            "seed_rank",
            "article_id",
        ],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    return sorted_variants.groupby(
        ["customer_id", "seed_article_id"],
        group_keys=False,
    ).head(per_seed_top_n)


def _rank_window_variants(variants: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if variants.empty:
        return pd.DataFrame(
            columns=["customer_id", "article_id", "source", "source_rank"]
        )

    deduped = (
        variants.groupby(["customer_id", "article_id"], as_index=False)
        .agg(
            article_popularity=("article_popularity", "max"),
            seed_rank=("seed_rank", "min"),
        )
        .sort_values(
            ["customer_id", "article_popularity", "seed_rank", "article_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
    )
    ranked = deduped.groupby("customer_id", group_keys=False).head(top_n).copy()
    ranked["source"] = "product_variant"
    ranked["source_rank"] = ranked.groupby("customer_id").cumcount() + 1
    return ranked.loc[:, ["customer_id", "article_id", "source", "source_rank"]]


def _attach_popularity(
    variants: pd.DataFrame,
    popularity: pd.DataFrame,
) -> pd.DataFrame:
    scored = variants.merge(popularity, on="article_id", how="left")
    scored["article_popularity"] = scored["article_popularity"].fillna(0).astype(int)
    return scored


def _concat_source_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return _empty_source_frame()
    result = pd.concat(non_empty, ignore_index=True)
    return _with_string_ids(result).loc[:, SOURCE_COLUMNS]


def _empty_source_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SOURCE_COLUMNS)


def _with_string_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in ("article_id", "customer_id", "source"):
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result
