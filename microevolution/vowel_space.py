"""Small, testable summaries used by the vowel-space review UI."""

from __future__ import annotations

import pandas as pd


def date_centroids(tokens: pd.DataFrame) -> pd.DataFrame:
    """Return robust (median) F1/F2 locations and token counts by date."""
    if tokens.empty:
        return pd.DataFrame(columns=["effective_date", "f1_hz", "f2_hz", "tokens"])
    return (
        tokens.groupby("effective_date", as_index=False)
        .agg(f1_hz=("f1_hz", "median"), f2_hz=("f2_hz", "median"), tokens=("token_id", "count"))
        .sort_values("effective_date")
        .reset_index(drop=True)
    )


def endpoint_change(centroids: pd.DataFrame) -> dict[str, object] | None:
    """Describe the first-to-last movement in acoustic and articulatory terms."""
    if len(centroids) < 2:
        return None
    first, last = centroids.iloc[0], centroids.iloc[-1]
    delta_f1 = float(last["f1_hz"] - first["f1_hz"])
    delta_f2 = float(last["f2_hz"] - first["f2_hz"])
    return {
        "first_date": pd.Timestamp(first["effective_date"]).date().isoformat(),
        "last_date": pd.Timestamp(last["effective_date"]).date().isoformat(),
        "first_tokens": int(first["tokens"]),
        "last_tokens": int(last["tokens"]),
        "delta_f1": delta_f1,
        "delta_f2": delta_f2,
        # In the conventional F1/F2 interpretation, higher F1 corresponds to a
        # lower tongue position and higher F2 to a fronter tongue position.
        "height": "lower" if delta_f1 > 0 else "higher" if delta_f1 < 0 else "unchanged",
        "backness": "fronter" if delta_f2 > 0 else "backer" if delta_f2 < 0 else "unchanged",
    }
