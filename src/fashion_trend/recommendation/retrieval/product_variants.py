from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fashion_trend.recommendation.retrieval.popularity import SOURCE_COLUMNS


@dataclass(frozen=True)
class ArticlePopularity:
    weekly_counts: pd.DataFrame
    article_code_by_id: pd.Series


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
    article_popularity = _prepare_article_popularity(transactions)
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
            article_popularity,
            cutoff_week=int(window["cutoff_week"]),
        )
        limited = _limit_variants_per_seed(
            variants,
            popularity,
            article_popularity,
            per_seed_top_n,
        )
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
    article_popularity: ArticlePopularity,
    cutoff_week: int,
) -> pd.Series:
    weekly_counts = article_popularity.weekly_counts
    history = weekly_counts.loc[weekly_counts["week_id"].le(cutoff_week)]
    if history.empty:
        return pd.Series(dtype="int64")
    return history.groupby("article_code", sort=False)["article_popularity"].sum()


def _limit_variants_per_seed(
    variants: pd.DataFrame,
    popularity: pd.Series,
    article_popularity: ArticlePopularity,
    per_seed_top_n: int,
) -> pd.DataFrame:
    scored = _attach_popularity(variants, popularity, article_popularity)
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
    popularity: pd.Series,
    article_popularity: ArticlePopularity,
) -> pd.DataFrame:
    scored = variants.copy()
    article_codes = pd.Series(
        article_popularity.article_code_by_id.reindex(
            scored["article_id"].to_numpy()
        ).to_numpy(),
        index=scored.index,
    )
    scored["article_popularity"] = 0
    known_articles = article_codes.notna()
    if known_articles.any():
        scored.loc[known_articles, "article_popularity"] = (
            popularity.reindex(article_codes.loc[known_articles].astype(np.int32))
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )
    return scored


def _prepare_article_popularity(transactions: pd.DataFrame) -> ArticlePopularity:
    if transactions.empty:
        return ArticlePopularity(
            weekly_counts=pd.DataFrame(
                columns=["week_id", "article_code", "article_popularity"]
            ),
            article_code_by_id=pd.Series(dtype=np.int32),
        )
    article_codes, article_ids = pd.factorize(transactions["article_id"], sort=False)
    article_ids = pd.Index(article_ids).astype(str)
    data = pd.DataFrame(
        {
            "week_id": pd.to_numeric(transactions["week_id"], errors="raise").astype(
                np.int16
            ),
            "article_code": article_codes.astype(np.int32, copy=False),
        }
    )
    weekly_counts = (
        data.groupby(["week_id", "article_code"], as_index=False, sort=False)
        .size()
        .rename(columns={"size": "article_popularity"})
    )
    article_code_by_id = pd.Series(
        np.arange(len(article_ids), dtype=np.int32),
        index=article_ids,
    )
    return ArticlePopularity(
        weekly_counts=weekly_counts,
        article_code_by_id=article_code_by_id,
    )


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
