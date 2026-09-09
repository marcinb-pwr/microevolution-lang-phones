"""Recording-aware stochastic comparison of temporal and constant models."""

from __future__ import annotations

from datetime import date
import random
from statistics import mean

from .project import Project, ProjectError


def _fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    xbar, ybar = mean(x), mean(y)
    denominator = sum((value - xbar) ** 2 for value in x)
    if denominator == 0:
        raise ProjectError("temporal model requires at least two distinct dates")
    slope = sum((a - xbar) * (b - ybar) for a, b in zip(x, y)) / denominator
    intercept = ybar - slope * xbar
    sse = sum((b - intercept - slope * a) ** 2 for a, b in zip(x, y))
    return intercept, slope, sse


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return ordered[lower] if lower + 1 == len(ordered) else (
        ordered[lower] * (1 - fraction) + ordered[lower + 1] * fraction
    )


def compare_models(project: Project, *, phone: str, response: str = "f1_hz",
                   iterations: int = 2000, seed: int = 0) -> dict:
    """Compare intercept-only and linear-time models by cluster permutation.

    Each recording contributes one mean, preventing recordings with many tokens
    from dominating.  The p-value permutes dates across recordings; the interval
    resamples recordings.  This is an exploratory comparison, not causal proof.
    """
    if response not in {"f1_hz", "f2_hz", "f0_hz", "duration_s"}:
        raise ProjectError(f"unsupported response: {response}")
    if iterations < 1:
        raise ProjectError("iterations must be positive")
    values: dict[str, list[float]] = {}
    for token in project.tokens:
        if token["phone_label"] == phone and token[response] and token["review_status"] != "rejected":
            values.setdefault(token["recording_id"], []).append(float(token[response]))
    rows = []
    for recording_id, measurements in values.items():
        recording = project.recording(recording_id)
        when = project.effective_date(recording)
        if when:
            rows.append((recording_id, when, mean(measurements), len(measurements)))
    if len(rows) < 3 or len({r[1] for r in rows}) < 2:
        raise ProjectError("model comparison requires at least three dated recordings and two dates")
    origin = min(r[1] for r in rows)
    x = [(r[1] - origin).days / 365.2425 for r in rows]
    y = [r[2] for r in rows]
    intercept, slope, temporal_sse = _fit(x, y)
    null_sse = sum((value - mean(y)) ** 2 for value in y)
    improvement = null_sse - temporal_sse
    rng = random.Random(seed)
    permuted = []
    slopes = []
    for _ in range(iterations):
        shuffled = rng.sample(x, len(x))
        permuted.append(null_sse - _fit(shuffled, y)[2])
        selected = [rng.randrange(len(rows)) for _ in rows]
        bx, by = [x[i] for i in selected], [y[i] for i in selected]
        if len(set(bx)) > 1:
            slopes.append(_fit(bx, by)[1])
    interval = [_quantile(slopes, 0.025), _quantile(slopes, 0.975)] if slopes else [None, None]
    return {"phone": phone, "response": response, "unit": "recording mean",
            "n_recordings": len(rows), "n_tokens": sum(r[3] for r in rows),
            "origin_date": date.isoformat(origin), "slope_per_year": slope,
            "slope_bootstrap_95pct": interval, "null_sse": null_sse,
            "temporal_sse": temporal_sse, "sse_improvement": improvement,
            "permutation_p_value": (1 + sum(v >= improvement for v in permuted)) / (iterations + 1),
            "iterations": iterations, "seed": seed,
            "caution": "Exploratory recording-level comparison; upload dates may be proxies."}
