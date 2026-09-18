"""Recording-aware stochastic comparison of temporal and constant models."""

from __future__ import annotations

from datetime import date
import random
from statistics import mean

from .project import Project, ProjectError, ReviewStore, merged_tokens, select_tokens


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
                   iterations: int = 2000, seed: int = 0,
                   reviews: ReviewStore | None = None, speaker_id: str | None = None,
                   data_origin: str = "observed", verified_only: bool = True,
                   include_pending: bool = False, automatic_qc: bool = False) -> dict:
    """Compare intercept-only and linear-time models by cluster permutation.

    Each recording contributes one mean, preventing recordings with many tokens
    from dominating.  The p-value permutes dates across recordings; the interval
    resamples recordings.  This is an exploratory comparison, not causal proof.
    """
    if response not in {"f1_hz", "f2_hz", "f0_hz", "duration_s"}:
        raise ProjectError(f"unsupported response: {response}")
    if iterations < 1:
        raise ProjectError("iterations must be positive")
    speakers = sorted({r["speaker_id"] for r in project.recordings})
    if speaker_id is None:
        if len(speakers) != 1:
            raise ProjectError("speaker_id is required when a project contains multiple speakers")
        speaker_id = speakers[0]
    reviews = reviews or ReviewStore(project.root / "reviews.sqlite3")
    requested = [t for t in merged_tokens(project, reviews)
                 if t["phone_label"] == phone and t["data_origin"] == data_origin
                 and project.recording(t["recording_id"])["speaker_id"] == speaker_id]
    if not requested:
        available = ", ".join(sorted({t["phone_label"] for t in project.tokens}))
        raise ProjectError(f"phone {phone!r} does not occur in the token table; available phones: {available}")
    values: dict[str, list[float]] = {}
    statuses = (("accepted", "pending") if (include_pending or automatic_qc)
                else ("accepted",))
    for token in select_tokens(project, reviews, speaker_id=speaker_id, phone=phone,
                               data_origin=data_origin, statuses=statuses,
                               verified_only=verified_only, automatic_qc=automatic_qc):
        if token[response]:
            values.setdefault(token["recording_id"], []).append(float(token[response]))
    rows = []
    for recording_id, measurements in values.items():
        recording = project.recording(recording_id)
        when = project.effective_date(recording)
        if when:
            rows.append((recording_id, when, mean(measurements), len(measurements)))
    if len(rows) < 3 or len({r[1] for r in rows}) < 2:
        status_counts = {}
        for token in requested:
            status = token["review_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        status_summary = ", ".join(
            f"{key}={value}" for key, value in sorted(status_counts.items()))
        raise ProjectError(
            "model comparison requires at least three dated recordings and two dates after "
            "eligibility filtering; "
            f"found {len(rows)} eligible recording(s) for {phone!r}. Token review statuses: "
            f"{status_summary or 'none'}. By default only accepted tokens with manually verified "
            "boundaries are eligible. Review tokens first, or, for an explicitly exploratory "
            "analysis of automatic output, pass both --include-pending and --include-unverified."
        )
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
            "included_review_statuses": list(statuses),
            "included_unverified_boundaries": not verified_only,
            "automatic_qc_eligibility": automatic_qc,
            "exclusions": _exclusion_counts(requested, values),
            "caution": "Exploratory recording-level comparison; upload dates may be proxies."}


def _exclusion_counts(requested, included_by_recording):
    counts = {}
    for token in requested:
        if token["recording_id"] in included_by_recording and token.get("measurement_status") == "measured":
            continue
        reason = (token.get("alignment_qc_reasons") or token.get("exclusion_reason")
                  or "not_eligible")
        counts[reason] = counts.get(reason, 0) + 1
    return counts
