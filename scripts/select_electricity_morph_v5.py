"""Build electricity_15min_morph v6 with topology-aware global diversity.

The selector operates on noise-free MA-65 trends.  It deliberately separates
an 8k core from a 22k multi-channel oversample pool. A final global radius
filter removes phase-aligned near duplicates and isolated jump artifacts before
noise is added. The final target is an upper bound: it is never filled with a
near duplicate merely to reach a fixed size.

Outputs (default: datasets/electricity_15min_morph):
  windows.npy, meta.csv, summary.json, DATASET_CARD.md,
  topology_grid.png, random_grid.png, nearest_pairs.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/public/home/liym2024/Verbalts_SAE")
DEFAULT_ARROW = Path(
    "/storage/group/renkan/TSFM_Datasets/chronos_datasets_arrow/"
    "electricity_15min/train-00000-of-00001.arrow"
)
DEFAULT_OUTPUT = ROOT / "datasets/electricity_15min_morph"

WINDOW = 128
MA_WIN = 65
CAND_STRIDE = 32
# TARGET controls the broad, channel-balanced oversample pool. FINAL_TARGET is
# deliberately an upper bound rather than a quota.
TARGET = 30_000
FINAL_TARGET = 12_000
CORE_TARGET = 8_000
CORE_CHANNEL_CAP = 30
TOTAL_CHANNEL_CAP = 100
CORE_MIN_GAP = 256
EXPANSION_MIN_GAP = 128
PROFILE_SIZE = 32
SHIFT_LIMIT = 32
NOISE_LEVEL = 0.10
N_PROTOTYPES = 512
DIVERSITY_DISTANCE = 0.06
MAX_PROFILE_JUMP = 1.50
MAX_JUMP_SHARE = 0.32
MAX_LOW_AMPLITUDE_RELATIVE_RANGE = 0.005
REPEATED_JUMP_FRACTION = 0.20
MAX_REPEATED_JUMPS = 19
SEED = 42

TOPOLOGIES = (
    "single_peak",
    "single_trough",
    "peak_trough",
    "trough_peak",
    "oscillatory",
    "step_up",
    "step_down",
    "monotonic_up",
    "monotonic_down",
    "flat_complex",
)


def smooth_series(values: np.ndarray, window: int = MA_WIN) -> np.ndarray:
    """Centered moving average with edge padding, matching prior versions."""
    values = np.asarray(values, dtype=np.float64)
    pad = window // 2
    return np.convolve(
        np.pad(values, (pad, pad), mode="edge"),
        np.ones(window, dtype=np.float64) / window,
        mode="valid",
    )


def paa(windows: np.ndarray, size: int = PROFILE_SIZE) -> np.ndarray:
    """Piecewise aggregate approximation (WINDOW must be divisible by size)."""
    return windows.reshape(len(windows), size, windows.shape[1] // size).mean(axis=2)


def robust_normalize(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q05, q95 = np.quantile(rows, [0.05, 0.95], axis=1)
    scale = np.maximum(q95 - q05, 1e-8)
    center = np.median(rows, axis=1)
    return ((rows - center[:, None]) / scale[:, None]).astype(np.float32), scale


def _merge_extrema(indices: np.ndarray, values: np.ndarray, min_distance: int = 4) -> np.ndarray:
    """Keep the stronger extremum when adjacent extrema are too close."""
    if len(indices) < 2:
        return indices
    order = indices[np.argsort(np.abs(values[indices]))[::-1]]
    kept: List[int] = []
    for idx in order:
        if all(abs(int(idx) - old) >= min_distance for old in kept):
            kept.append(int(idx))
    return np.asarray(sorted(kept), dtype=np.int16)


def classify_profile(profile: np.ndarray, prominence: float = 0.10) -> Tuple[str, int, int]:
    """Classify one robust-normalized PAA profile and return topology/phase/extrema.

    Prominence is relative to robust range (normalization makes it approximately
    one).  Extrema near an edge are ignored because a truncated daily cycle is
    better represented as monotonic/step-like than as a complete peak/valley.
    """
    p = np.asarray(profile, dtype=np.float64)
    # A short [1,2,1] filter prevents one-bin numerical wiggles becoming turns.
    s = np.convolve(np.pad(p, (1, 1), mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    d = np.diff(s)
    maxima = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    minima = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    maxima = maxima[(maxima >= 3) & (maxima <= len(p) - 4)]
    minima = minima[(minima >= 3) & (minima <= len(p) - 4)]
    baseline = 0.5 * (s[0] + s[-1])
    maxima = maxima[s[maxima] - baseline >= prominence]
    minima = minima[baseline - s[minima] >= prominence]
    maxima = _merge_extrema(maxima, s - baseline)
    minima = _merge_extrema(minima, s - baseline)

    extrema = sorted([(int(i), "p", abs(s[i] - baseline)) for i in maxima] +
                     [(int(i), "t", abs(s[i] - baseline)) for i in minima])
    n_extrema = len(extrema)
    endpoint_change = s[-3:].mean() - s[:3].mean()
    consistency = abs(np.mean(np.sign(d)))
    abs_d = np.abs(d)
    jump = int(np.argmax(abs_d))
    jump_share = float(abs_d[jump] / (abs_d.sum() + 1e-12))
    left_flat = np.std(d[:max(2, jump)]) < 0.045
    right_flat = np.std(d[min(jump + 1, len(d) - 2):]) < 0.045

    if n_extrema >= 3:
        topology = "oscillatory"
        main = max(extrema, key=lambda x: x[2])[0]
    elif len(maxima) == 1 and len(minima) == 0:
        topology, main = "single_peak", int(maxima[0])
    elif len(minima) == 1 and len(maxima) == 0:
        topology, main = "single_trough", int(minima[0])
    elif maxima.size and minima.size:
        first_peak, first_trough = int(maxima[0]), int(minima[0])
        topology = "peak_trough" if first_peak < first_trough else "trough_peak"
        main = max(extrema, key=lambda x: x[2])[0]
    elif jump_share >= 0.18 and left_flat and right_flat and abs(endpoint_change) >= 0.25:
        topology = "step_up" if endpoint_change > 0 else "step_down"
        main = jump
    elif endpoint_change >= 0.20 and consistency >= 0.20:
        topology, main = "monotonic_up", len(p) - 1
    elif endpoint_change <= -0.20 and consistency >= 0.20:
        topology, main = "monotonic_down", 0
    else:
        topology = "flat_complex"
        main = int(np.argmax(np.abs(s - baseline)))

    phase_bin = min(2, (3 * main) // len(p))
    return topology, int(phase_bin), n_extrema


def align_profile(profile: np.ndarray, topology: str, phase_bin: int) -> np.ndarray:
    """Canonicalize phase using edge extension rather than circular wrapping."""
    p = np.asarray(profile, dtype=np.float32)
    if topology in {"monotonic_up", "monotonic_down", "flat_complex"}:
        return p.copy()
    smooth = np.convolve(np.pad(p, (1, 1), mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    if topology in {"single_peak", "peak_trough"}:
        anchor = int(np.argmax(smooth))
    elif topology in {"single_trough", "trough_peak"}:
        anchor = int(np.argmin(smooth))
    else:
        anchor = int(np.argmax(np.abs(smooth - np.median(smooth))))
    source = np.arange(len(p), dtype=np.float32) + anchor - len(p) // 2
    return np.interp(source, np.arange(len(p)), p, left=p[0], right=p[-1]).astype(np.float32)


def shift_correlation(a: np.ndarray, b: np.ndarray, max_shift: int = SHIFT_LIMIT) -> float:
    """Maximum Pearson correlation over non-circular shifts."""
    best = -1.0
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            x, y = a[-shift:], b[:shift]
        elif shift > 0:
            x, y = a[:-shift], b[shift:]
        else:
            x, y = a, b
        if len(x) < 8 or x.std() < 1e-10 or y.std() < 1e-10:
            continue
        best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return best


def _balanced_targets(total: int, capacities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Water-fill an integer target as evenly as capacities permit."""
    targets = np.zeros(len(capacities), dtype=np.int32)
    order = rng.permutation(len(capacities))
    while targets.sum() < total:
        changed = False
        for i in order:
            if targets[i] < capacities[i]:
                targets[i] += 1
                changed = True
                if targets.sum() == total:
                    break
        if not changed:
            raise RuntimeError(f"Only {targets.sum()} samples satisfy channel capacities; need {total}")
    return targets


def _temporal_ok(start: int, selected_starts: List[int], gap: int) -> bool:
    return all(abs(start - old) >= gap for old in selected_starts)


def _select_channel(
    members: np.ndarray,
    starts: np.ndarray,
    topology_ids: np.ndarray,
    phases: np.ndarray,
    prototypes: np.ndarray,
    features: np.ndarray,
    need: int,
    gap: int,
    already: Sequence[int],
    global_topology: Counter,
    global_phase: Counter,
    global_prototype: Counter,
    rng: np.random.Generator,
) -> List[int]:
    """Channel-local farthest-point selection with global rarity rewards."""
    chosen: List[int] = []
    selected_starts = [int(starts[i]) for i in already]
    pool = members.copy()
    rng.shuffle(pool)
    # Compress very long channels while retaining each topology/phase cell.
    if len(pool) > 2500:
        cell_keep: List[int] = []
        for t in np.unique(topology_ids[pool]):
            for p in range(3):
                cell = pool[(topology_ids[pool] == t) & (phases[pool] == p)]
                if len(cell):
                    cell_keep.extend(cell[: min(250, len(cell))])
        pool = np.asarray(cell_keep, dtype=np.int64)

    min_dist = np.full(len(pool), 2.0, dtype=np.float32)
    if already:
        base = features[np.asarray(already)]
        for j in range(0, len(pool), 512):
            block = features[pool[j:j + 512]]
            min_dist[j:j + 512] = np.sqrt(
                ((block[:, None] - base[None]) ** 2).mean(axis=2)
            ).min(axis=1)

    active = np.ones(len(pool), dtype=bool)
    while len(chosen) < need:
        valid_positions = np.where(active)[0]
        if not len(valid_positions):
            break
        candidates = pool[valid_positions]
        temporal = np.asarray([
            _temporal_ok(int(starts[i]), selected_starts, gap) for i in candidates
        ])
        valid_positions = valid_positions[temporal]
        if not len(valid_positions):
            break
        candidates = pool[valid_positions]
        topo_bonus = np.asarray([1.0 / math.sqrt(1 + global_topology[int(topology_ids[i])])
                                 for i in candidates])
        phase_bonus = np.asarray([1.0 / math.sqrt(1 + global_phase[(int(topology_ids[i]), int(phases[i]))])
                                  for i in candidates])
        proto_bonus = np.asarray([1.0 / math.sqrt(1 + global_prototype[int(prototypes[i])])
                                  for i in candidates])
        score = min_dist[valid_positions] + 0.55 * topo_bonus + 0.25 * phase_bonus + 0.65 * proto_bonus
        pos = int(valid_positions[int(np.argmax(score))])
        idx = int(pool[pos])
        chosen.append(idx)
        selected_starts.append(int(starts[idx]))
        global_topology[int(topology_ids[idx])] += 1
        global_phase[(int(topology_ids[idx]), int(phases[idx]))] += 1
        global_prototype[int(prototypes[idx])] += 1
        active[pos] = False
        distance = np.sqrt(((features[pool] - features[idx]) ** 2).mean(axis=1))
        min_dist = np.minimum(min_dist, distance)
    return chosen


def allocate_prototypes(counts: np.ndarray, total: int = N_PROTOTYPES) -> np.ndarray:
    """Allocate prototypes sub-linearly so dense topologies cannot monopolize."""
    nonzero = counts > 0
    allocation = np.zeros_like(counts, dtype=np.int32)
    allocation[nonzero] = 4
    remaining = total - int(allocation.sum())
    weights = np.sqrt(counts.astype(np.float64)) * nonzero
    raw = remaining * weights / weights.sum()
    allocation += np.floor(raw).astype(np.int32)
    for i in np.argsort(raw - np.floor(raw))[::-1][: total - int(allocation.sum())]:
        allocation[i] += 1
    allocation = np.minimum(allocation, counts)
    return allocation


def build_candidates(table) -> Dict[str, np.ndarray]:
    """Extract candidate descriptors without retaining the large raw-window pool."""
    ids = table.column("id").to_pylist()
    values = table.column("consumption_kW")
    profiles: List[np.ndarray] = []
    canonical: List[np.ndarray] = []
    channel_indices: List[np.ndarray] = []
    starts_all: List[np.ndarray] = []
    topology_all: List[np.ndarray] = []
    phase_all: List[np.ndarray] = []
    extrema_all: List[np.ndarray] = []
    relative_range_all: List[np.ndarray] = []
    repeated_jumps_all: List[np.ndarray] = []
    channel_ids: List[str] = []

    for ci, cid in enumerate(ids):
        raw = np.asarray(values[ci].as_py(), dtype=np.float64)
        raw = raw[np.isfinite(raw)]
        if len(raw) < WINDOW:
            continue
        trend = smooth_series(raw)
        starts = np.arange(0, len(trend) - WINDOW + 1, CAND_STRIDE, dtype=np.int32)
        windows = np.lib.stride_tricks.sliding_window_view(trend, WINDOW)[::CAND_STRIDE]
        prof, scales = robust_normalize(paa(windows))
        # Constant windows carry no morphology and destabilize correlation metrics.
        valid = scales > max(1e-7, float(np.median(scales)) * 1e-4)
        window_q05, window_q95 = np.quantile(windows, [0.05, 0.95], axis=1)
        window_range = np.maximum(window_q95 - window_q05, 1e-12)
        relative_range = window_range / np.maximum(np.abs(np.median(windows, axis=1)), 1e-8)
        repeated_jumps = (
            np.abs(np.diff(windows, axis=1))
            > REPEATED_JUMP_FRACTION * window_range[:, None]
        ).sum(axis=1)
        prof, starts = prof[valid], starts[valid]
        relative_range, repeated_jumps = relative_range[valid], repeated_jumps[valid]
        labels, phases, extrema = [], [], []
        aligned = np.empty_like(prof)
        for j, p in enumerate(prof):
            label, phase, n_extrema = classify_profile(p)
            labels.append(TOPOLOGIES.index(label))
            phases.append(phase)
            extrema.append(n_extrema)
            aligned[j] = align_profile(p, label, phase)
        profiles.append(prof)
        canonical.append(aligned)
        channel_indices.append(np.full(len(prof), len(channel_ids), dtype=np.int16))
        starts_all.append(starts)
        topology_all.append(np.asarray(labels, dtype=np.int8))
        phase_all.append(np.asarray(phases, dtype=np.int8))
        extrema_all.append(np.asarray(extrema, dtype=np.int8))
        relative_range_all.append(relative_range.astype(np.float32))
        repeated_jumps_all.append(repeated_jumps.astype(np.int16))
        channel_ids.append(str(cid))
        if (ci + 1) % 25 == 0 or ci + 1 == len(ids):
            print(f"candidate extraction: {ci + 1}/{len(ids)} channels")

    return {
        "channel_ids": np.asarray(channel_ids),
        "profile": np.concatenate(profiles),
        "canonical": np.concatenate(canonical),
        "channel": np.concatenate(channel_indices),
        "start": np.concatenate(starts_all),
        "topology": np.concatenate(topology_all),
        "phase": np.concatenate(phase_all),
        "n_extrema": np.concatenate(extrema_all),
        "relative_range": np.concatenate(relative_range_all),
        "repeated_jumps": np.concatenate(repeated_jumps_all),
    }


def fit_prototypes(candidates: Dict[str, np.ndarray], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import MiniBatchKMeans

    labels = candidates["topology"]
    features = candidates["canonical"]
    counts = np.bincount(labels, minlength=len(TOPOLOGIES))
    allocation = allocate_prototypes(counts)
    assignments = np.empty(len(labels), dtype=np.int32)
    centers: List[np.ndarray] = []
    offset = 0
    for topology_id, n_clusters in enumerate(allocation):
        members = np.where(labels == topology_id)[0]
        if not len(members):
            continue
        fit_idx = members if len(members) <= 50_000 else rng.choice(members, 50_000, replace=False)
        model = MiniBatchKMeans(
            n_clusters=int(n_clusters), batch_size=4096, n_init=3,
            random_state=SEED + topology_id, max_iter=100,
        ).fit(features[fit_idx])
        assignments[members] = model.predict(features[members]) + offset
        centers.extend(model.cluster_centers_)
        offset += int(n_clusters)
        print(f"prototypes: {TOPOLOGIES[topology_id]} n={len(members):,}, k={n_clusters}")
    return assignments, np.asarray(centers, dtype=np.float32)


def select(candidates: Dict[str, np.ndarray], prototypes: np.ndarray, rng: np.random.Generator):
    channel = candidates["channel"]
    starts = candidates["start"]
    topology = candidates["topology"]
    phases = candidates["phase"]
    features = candidates["canonical"]
    n_channels = len(candidates["channel_ids"])
    capacities = np.minimum(TOTAL_CHANNEL_CAP, np.bincount(channel, minlength=n_channels))
    total_targets = _balanced_targets(TARGET, capacities, rng)
    core_targets = _balanced_targets(
        CORE_TARGET, np.minimum(np.minimum(CORE_CHANNEL_CAP, capacities), total_targets), rng
    )
    global_topology: Counter = Counter()
    global_phase: Counter = Counter()
    global_prototype: Counter = Counter()
    selected_by_channel: List[List[int]] = [[] for _ in range(n_channels)]
    tiers: Dict[int, str] = {}

    for tier, targets, gap in (
        ("core", core_targets, CORE_MIN_GAP),
        ("expansion", total_targets - core_targets, EXPANSION_MIN_GAP),
    ):
        for ci in rng.permutation(n_channels):
            need = int(targets[ci])
            members = np.where(channel == ci)[0]
            picked = _select_channel(
                members, starts, topology, phases, prototypes, features, need, gap,
                selected_by_channel[ci], global_topology, global_phase,
                global_prototype, rng,
            )
            # Rare short channels may not satisfy the desired gap. Relax only enough
            # to retain channel balance, and record the effective tier globally.
            if len(picked) < need:
                extra = _select_channel(
                    members, starts, topology, phases, prototypes, features,
                    need - len(picked), CAND_STRIDE,
                    selected_by_channel[ci] + picked, global_topology,
                    global_phase, global_prototype, rng,
                )
                picked.extend(extra)
            if len(picked) != need:
                raise RuntimeError(f"channel {ci}: selected {len(picked)}/{need} for {tier}")
            selected_by_channel[ci].extend(picked)
            tiers.update({idx: tier for idx in picked})
        print(f"selected {tier}: {sum(v == tier for v in tiers.values()):,}")

    selected = np.asarray([i for group in selected_by_channel for i in group], dtype=np.int64)
    return selected, np.asarray([tiers[int(i)] for i in selected]), core_targets, total_targets


def diversity_prune(
    selected: np.ndarray,
    tiers: np.ndarray,
    candidates: Dict[str, np.ndarray],
    prototypes: np.ndarray,
    limit: int = FINAL_TARGET,
    radius: float = DIVERSITY_DISTANCE,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Globally suppress near duplicates in the channel-balanced oversample pool.

    Distances are RMS distances between phase-aligned, robust-normalized PAA-32
    profiles. Core candidates are considered before expansion candidates, but a
    first pass gives every source channel an opportunity to contribute. The
    result is a maximal radius-separated subset (up to ``limit``), not a
    density-weighted sample.
    """
    from sklearn.neighbors import NearestNeighbors

    features = candidates["canonical"][selected]
    topology = candidates["topology"][selected]
    channels = candidates["channel"][selected]
    profile_diff = np.abs(np.diff(features, axis=1))
    max_jump = profile_diff.max(axis=1)
    jump_share = max_jump / np.maximum(profile_diff.sum(axis=1), 1e-8)
    step_ids = {TOPOLOGIES.index("step_up"), TOPOLOGIES.index("step_down")}
    is_step = np.isin(topology, list(step_ids))
    repeated_pulse = (
        (candidates["relative_range"][selected] <= MAX_LOW_AMPLITUDE_RELATIVE_RANGE)
        & (candidates["repeated_jumps"][selected] > MAX_REPEATED_JUMPS)
    )
    quality = (
        (max_jump <= MAX_PROFILE_JUMP)
        & (is_step | (jump_share <= MAX_JUMP_SHARE))
        & ~repeated_pulse
    )

    # Radius-neighbor lists make suppression linear in the actual number of
    # near-duplicate links instead of quadratic in the candidate count.
    neighbors: List[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in selected]
    euclidean_radius = radius * math.sqrt(features.shape[1])
    local_novelty = np.full(len(selected), np.inf, dtype=np.float32)
    for topology_id in np.unique(topology):
        positions = np.flatnonzero((topology == topology_id) & quality)
        if not len(positions):
            continue
        model = NearestNeighbors(
            radius=euclidean_radius, n_neighbors=min(2, len(positions))
        ).fit(features[positions])
        local = model.radius_neighbors(features[positions], return_distance=False)
        for pos, linked in zip(positions, local):
            neighbors[int(pos)] = positions[np.asarray(linked, dtype=np.int64)].astype(np.int32)
        if len(positions) > 1:
            distances, _ = model.kneighbors(features[positions])
            local_novelty[positions] = distances[:, 1] / math.sqrt(features.shape[1])

    prototype_frequency = Counter(int(prototypes[idx]) for idx in selected[quality])
    rng_tie = np.random.default_rng(SEED + 17).random(len(selected))
    eligible = np.flatnonzero(quality)
    ranked = sorted(
        eligible,
        key=lambda pos: (
            0 if tiers[pos] == "core" else 1,
            prototype_frequency[int(prototypes[selected[pos]])],
            -float(local_novelty[pos]),
            float(rng_tie[pos]),
        ),
    )

    blocked = np.zeros(len(selected), dtype=bool)
    kept: List[int] = []

    def accept(pos: int) -> None:
        kept.append(pos)
        blocked[neighbors[pos]] = True

    # Channel-first seeding is the explicit "look in other channels" step.
    by_channel: Dict[int, List[int]] = {}
    for pos in ranked:
        by_channel.setdefault(int(channels[pos]), []).append(pos)
    for channel_id in sorted(by_channel):
        for pos in by_channel[channel_id]:
            if not blocked[pos]:
                accept(pos)
                break
        if len(kept) >= limit:
            break
    if len(kept) < limit:
        for pos in ranked:
            if not blocked[pos]:
                accept(pos)
                if len(kept) >= limit:
                    break

    kept_pos = np.asarray(kept, dtype=np.int64)
    stats = {
        "oversample_windows": int(len(selected)),
        "quality_rejected": int((~quality).sum()),
        "repeated_low_amplitude_pulse_rejected": int(repeated_pulse.sum()),
        "near_duplicate_or_limit_rejected": int(quality.sum() - len(kept_pos)),
        "diversity_distance_rms": radius,
        "final_target_upper_bound": limit,
    }
    return selected[kept_pos], tiers[kept_pos], stats


def reconstruct_windows(table, candidates: Dict[str, np.ndarray], selected: np.ndarray) -> np.ndarray:
    ids = table.column("id").to_pylist()
    id_to_row = {str(cid): i for i, cid in enumerate(ids)}
    output = np.empty((len(selected), WINDOW), dtype=np.float32)
    selected_channels = candidates["channel"][selected]
    for ci, cid in enumerate(candidates["channel_ids"]):
        positions = np.where(selected_channels == ci)[0]
        if not len(positions):
            continue
        raw = np.asarray(table.column("consumption_kW")[id_to_row[str(cid)]].as_py(), dtype=np.float64)
        trend = smooth_series(raw[np.isfinite(raw)])
        for pos in positions:
            start = int(candidates["start"][selected[pos]])
            output[pos] = trend[start:start + WINDOW]
    return output


def nearest_core_metrics(
    selected: np.ndarray, tiers: np.ndarray, features: np.ndarray, topology: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    distances = np.full(len(selected), np.nan, dtype=np.float32)
    neighbors = np.full(len(selected), -1, dtype=np.int32)
    core_pos = np.where(tiers == "core")[0]
    for t in np.unique(topology[selected]):
        query_pos = np.where(topology[selected] == t)[0]
        reference_pos = core_pos[topology[selected[core_pos]] == t]
        if not len(reference_pos):
            continue
        model = NearestNeighbors(n_neighbors=min(2, len(reference_pos))).fit(features[selected[reference_pos]])
        d, nn = model.kneighbors(features[selected[query_pos]])
        for row, pos in enumerate(query_pos):
            rank = 1 if tiers[pos] == "core" and d.shape[1] > 1 else 0
            distances[pos] = d[row, rank] / math.sqrt(features.shape[1])
            neighbors[pos] = int(reference_pos[nn[row, rank]])
    return distances, neighbors


def save_plots(out: Path, windows: np.ndarray, rows: List[dict], neighbors: np.ndarray, rng):
    topologies = np.asarray([r["topology"] for r in rows])
    fig, axes = plt.subplots(len(TOPOLOGIES), 10, figsize=(20, 18), constrained_layout=True)
    for ti, name in enumerate(TOPOLOGIES):
        members = np.where(topologies == name)[0]
        show = members if len(members) <= 10 else rng.choice(members, 10, replace=False)
        for col, idx in enumerate(show):
            axes[ti, col].plot(windows[idx], lw=0.7)
            axes[ti, col].set_xticks([])
        axes[ti, 0].set_ylabel(f"{name}\n(n={len(members)})", fontsize=8)
        for col in range(len(show), 10):
            axes[ti, col].axis("off")
    fig.suptitle("electricity_15min_morph v6 — topology-stratified examples")
    fig.savefig(out / "topology_grid.png", dpi=120)
    plt.close(fig)

    pick = rng.choice(len(windows), min(100, len(windows)), replace=False)
    fig, axes = plt.subplots(10, 10, figsize=(20, 16), constrained_layout=True)
    for ax, idx in zip(axes.ravel(), pick):
        ax.plot(windows[idx], lw=0.7)
        ax.set_title(f"{rows[idx]['channel_id']} {rows[idx]['topology']}", fontsize=6)
        ax.set_xticks([])
    fig.savefig(out / "random_grid.png", dpi=120)
    plt.close(fig)

    expansion = np.where(np.asarray([r["selection_tier"] for r in rows]) == "expansion")[0]
    show = expansion[np.argsort([float(rows[i]["nearest_core_distance"]) for i in expansion])[:20]]
    fig, axes = plt.subplots(10, 2, figsize=(12, 20), constrained_layout=True)
    for row_idx, idx in enumerate(show[:10]):
        nn = neighbors[idx]
        axes[row_idx, 0].plot(windows[idx], lw=0.8)
        axes[row_idx, 1].plot(windows[nn], lw=0.8)
        axes[row_idx, 0].set_ylabel(f"d={rows[idx]['nearest_core_distance']}", fontsize=7)
        axes[row_idx, 0].set_title("expansion", fontsize=8)
        axes[row_idx, 1].set_title("nearest core", fontsize=8)
    fig.savefig(out / "nearest_pairs.png", dpi=120)
    plt.close(fig)


def build_summary(rows: List[dict], candidates: Dict[str, np.ndarray], prototypes: np.ndarray,
                  nearest_distances: np.ndarray, noisy: np.ndarray, selection_stats: dict) -> dict:
    channel_counts = Counter(r["channel_id"] for r in rows)
    topology_counts = Counter(r["topology"] for r in rows)
    phase_counts = Counter(f"{r['topology']}:{r['phase_bin']}" for r in rows)
    prototype_counts = Counter(int(r["prototype_id"]) for r in rows)
    return {
        "name": "electricity_15min_morph",
        "version": 6,
        "source_dataset": "chronos electricity_15min (train split)",
        "pipeline": {
            "smoothing": "centered moving average, window 65, edge-padded",
            "window_length": WINDOW,
            "candidate_stride": CAND_STRIDE,
            "topology_prominence": "0.10 * robust range",
            "phase_bins": 3,
            "canonical_profile_points": PROFILE_SIZE,
            "prototypes": int(len(set(prototypes.tolist()))),
            "core_target": CORE_TARGET,
            "core_channel_cap": CORE_CHANNEL_CAP,
            "total_channel_cap": TOTAL_CHANNEL_CAP,
            "core_min_start_gap": CORE_MIN_GAP,
            "expansion_min_start_gap": EXPANSION_MIN_GAP,
            "global_diversity_distance_rms": DIVERSITY_DISTANCE,
            "final_target_upper_bound": FINAL_TARGET,
            "max_profile_jump": MAX_PROFILE_JUMP,
            "max_jump_share": MAX_JUMP_SHARE,
            "max_low_amplitude_relative_range": MAX_LOW_AMPLITUDE_RELATIVE_RANGE,
            "repeated_jump_fraction": REPEATED_JUMP_FRACTION,
            "max_repeated_jumps": MAX_REPEATED_JUMPS,
        },
        "candidate_windows": int(len(candidates["start"])),
        "total_windows": len(rows),
        "channels": len(channel_counts),
        "channel_count_min_median_max": [min(channel_counts.values()),
                                          float(np.median(list(channel_counts.values()))),
                                          max(channel_counts.values())],
        "selection_tiers": dict(Counter(r["selection_tier"] for r in rows)),
        "selection_filter": selection_stats,
        "topology_counts": dict(topology_counts),
        "topology_phase_counts": dict(phase_counts),
        "effective_prototypes": len(prototype_counts),
        "max_prototype_count": max(prototype_counts.values()),
        "max_prototype_fraction": max(prototype_counts.values()) / len(rows),
        "nearest_core_distance": {
            "median": float(np.nanmedian(nearest_distances)),
            "p05": float(np.nanquantile(nearest_distances, 0.05)),
            "p95": float(np.nanquantile(nearest_distances, 0.95)),
        },
        "noise": {"type": "gaussian", "level": NOISE_LEVEL, "seed": SEED},
        "shape": list(noisy.shape),
        "dtype": str(noisy.dtype),
    }


def write_card(out: Path, summary: dict):
    counts = "\n".join(f"- `{k}`: {v:,}" for k, v in summary["topology_counts"].items())
    text = f"""# electricity_15min_morph v6

基于 Chronos `electricity_15min` 全部 {summary['channels']} 个有效通道重新筛选。
数据包含 {summary['total_windows']:,} 个长度 128 的 float32 窗口。

## 筛选流程

1. 对原序列做中心 MA-65，仅在无噪趋势上按 stride 32 构造候选。
2. 用相对 prominence（robust range 的 0.10）提取峰谷拓扑，并标记 left/center/right 相位。
3. 将主转折对齐后聚类为 prototype；prototype 仅用于候选压缩和覆盖，不按密度抽样。
4. 先从全部通道选 {CORE_TARGET:,} 条 core，再均衡扩展为 {TARGET:,} 条多通道候选池。
5. 在同拓扑内执行全局形态半径抑制：相位对齐后的 PAA-32 RMS 距离必须至少为
   {DIVERSITY_DISTANCE}；最终 {FINAL_TARGET:,} 是上限而不是必须填满的配额。先让每个通道
   贡献一个合格代表，不足部分再按 prototype 稀缺性和局部新颖度补齐。
6. 剔除孤立跳变轮廓（非 step 的最大一阶差分占比 > {MAX_JUMP_SHARE}，或最大差分 >
   {MAX_PROFILE_JUMP}），以及低振幅重复方波/窄脉冲（relative range ≤
   {MAX_LOW_AMPLITUDE_RELATIVE_RANGE} 且超过 {MAX_REPEATED_JUMPS} 次相邻跳变大于窗口
   robust range 的 {REPEATED_JUMP_FRACTION}）。选定后添加 `0.10 × per-window std` 高斯噪声。

## 拓扑分布

{counts}

## 文件和字段

- `windows.npy`: `[{summary['total_windows']}, 128]`，保留原始 kW 幅度。
- `meta.csv`: 含 `selection_tier`、`prototype_id`、`topology`、`phase_bin`、
  `nearest_core_distance`，以及来源通道和起点。
- `summary.json`: 通道、拓扑、prototype 和最近 core 距离验收指标。
- `validation.json`: 对最终产物独立执行的形状、间隔、覆盖和近邻验收结果。
- `topology_grid.png` / `random_grid.png` / `nearest_pairs.png`: 人工抽查图。

`nearest_core_distance` 是相位对齐后 32 点 robust-normalized 轮廓的 RMS 距离。
它用于快速全量验收；`shift_correlation` 则用于近邻图中的严格抽查。
"""
    (out / "DATASET_CARD.md").write_text(text)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise SystemExit("pyarrow is required; run this script in the project data environment") from exc

    rng = np.random.default_rng(SEED)
    table = pa.ipc.open_stream(str(args.arrow)).read_all()
    candidates = build_candidates(table)
    prototype_ids, centers = fit_prototypes(candidates, rng)
    selected, tiers, _, _ = select(candidates, prototype_ids, rng)
    if len(selected) != TARGET:
        raise RuntimeError(f"expected {TARGET} oversample windows, got {len(selected)}")
    selected, tiers, selection_stats = diversity_prune(
        selected, tiers, candidates, prototype_ids
    )
    print(f"global diversity filter: {TARGET:,} -> {len(selected):,}")
    clean = reconstruct_windows(table, candidates, selected)
    distances, neighbors = nearest_core_metrics(
        selected, tiers, candidates["canonical"], candidates["topology"]
    )
    noise = rng.standard_normal(clean.shape, dtype=np.float32)
    noisy = (clean + NOISE_LEVEL * clean.std(axis=1, keepdims=True) * noise).astype(np.float32)

    rows: List[dict] = []
    for pos, idx in enumerate(selected):
        rows.append({
            "dataset": "electricity_15min",
            "channel_id": str(candidates["channel_ids"][candidates["channel"][idx]]),
            "channel_index": int(candidates["channel"][idx]),
            "window_index": pos,
            "start_index": int(candidates["start"][idx]),
            "stride": CAND_STRIDE,
            "selection_tier": str(tiers[pos]),
            "prototype_id": int(prototype_ids[idx]),
            "topology": TOPOLOGIES[int(candidates["topology"][idx])],
            "phase_bin": ("left", "center", "right")[int(candidates["phase"][idx])],
            "n_significant_extrema": int(candidates["n_extrema"][idx]),
            "relative_range": round(float(candidates["relative_range"][idx]), 9),
            "repeated_jumps": int(candidates["repeated_jumps"][idx]),
            "nearest_core_distance": round(float(distances[pos]), 7),
            "noise_level": NOISE_LEVEL,
            "window_mean": float(noisy[pos].mean()),
            "window_std": float(noisy[pos].std()),
        })

    tmp = args.output.with_name(args.output.name + ".v6_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    np.save(tmp / "windows.npy", noisy)
    with (tmp / "meta.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = build_summary(
        rows, candidates, prototype_ids[selected], distances, noisy, selection_stats
    )
    (tmp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_card(tmp, summary)
    save_plots(tmp, noisy, rows, neighbors, rng)

    backup = args.output.with_name(args.output.name + "_pre_pulse_filter")
    if args.output.exists() and not backup.exists():
        args.output.rename(backup)
    elif args.output.exists():
        shutil.rmtree(args.output)
    tmp.rename(args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved -> {args.output}; previous dataset -> {backup}")


if __name__ == "__main__":
    main()