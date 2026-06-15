#!/usr/bin/env python3
"""
DRM scheduler ftrace analysis and plotting.

Parses ftrace output from gpu_scheduler tracepoints, matches job lifecycle
events (queue -> run -> done), computes per-entity latency statistics,
and optionally generates plots.
"""

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field


@dataclass
class JobRecord:
    """Complete lifecycle of a single job."""
    fence_ctx: int
    fence_seqno: int
    client_id: int
    ring: str
    queue_us: float = 0.0
    run_us: float = 0.0
    done_us: float = 0.0

    @property
    def queue_latency_us(self) -> float:
        if self.run_us and self.queue_us:
            return self.run_us - self.queue_us
        return 0.0

    @property
    def exec_time_us(self) -> float:
        if self.done_us and self.run_us:
            return self.done_us - self.run_us
        return 0.0

    @property
    def total_latency_us(self) -> float:
        if self.done_us and self.queue_us:
            return self.done_us - self.queue_us
        return 0.0


@dataclass
class EntityStats:
    """Per-entity aggregate statistics."""
    client_id: int
    ring: str
    job_count: int = 0
    total_exec_us: float = 0.0
    queue_latencies: list = field(default_factory=list)
    exec_times: list = field(default_factory=list)
    total_latencies: list = field(default_factory=list)
    queue_times: list = field(default_factory=list)  # absolute queue ts (us)


# ── Trace parsing ─────────────────────────────────────────────────────────

TRACE_LINE_RE = re.compile(
    r'\s*\S+-\d+\s+\[\d+\]\s+\S+\s+'
    r'(\d+\.\d+):\s+'
    r'(\w+):\s+'
    r'(.*)'
)

JOB_QUEUE_RE = re.compile(
    r'dev=(\S+),\s+fence=(\d+):(\d+),\s+ring=(\S+),\s+'
    r'job count:(\d+),\s+hw job count:(\d+),\s+client_id:(\d+)'
)

JOB_RUN_RE = JOB_QUEUE_RE

JOB_DONE_RE = re.compile(
    r'fence=(\d+):(\d+)\s+signaled'
)

def parse_trace_file(filepath: str) -> dict[tuple[int, int], JobRecord]:
    """Parse an ftrace file and return completed job records."""
    jobs: dict[tuple[int, int], JobRecord] = {}

    with open(filepath) as f:
        for line in f:
            m = TRACE_LINE_RE.match(line)
            if not m:
                continue

            ts_str, event, data = m.groups()
            ts_us = float(ts_str) * 1_000_000

            if event == "drm_sched_job_queue":
                dm = JOB_QUEUE_RE.match(data)
                if not dm:
                    continue
                ctx, seqno = int(dm.group(2)), int(dm.group(3))
                key = (ctx, seqno)
                jobs[key] = JobRecord(
                    fence_ctx=ctx,
                    fence_seqno=seqno,
                    client_id=int(dm.group(7)),
                    ring=dm.group(4),
                    queue_us=ts_us,
                )

            elif event == "drm_sched_job_run":
                dm = JOB_RUN_RE.match(data)
                if not dm:
                    continue
                key = (int(dm.group(2)), int(dm.group(3)))
                if key in jobs:
                    jobs[key].run_us = ts_us

            elif event == "drm_sched_job_done":
                dm = JOB_DONE_RE.match(data)
                if not dm:
                    continue
                key = (int(dm.group(1)), int(dm.group(2)))
                if key in jobs:
                    jobs[key].done_us = ts_us

    return jobs


def compute_entity_stats(jobs: dict) -> dict[int, EntityStats]:
    """Compute per-entity statistics from job records."""
    entities: dict[int, EntityStats] = {}

    for job in jobs.values():
        if not (job.queue_us and job.run_us and job.done_us):
            continue

        cid = job.client_id
        if cid not in entities:
            entities[cid] = EntityStats(client_id=cid, ring=job.ring)

        e = entities[cid]
        e.job_count += 1
        e.total_exec_us += job.exec_time_us
        e.queue_latencies.append(job.queue_latency_us)
        e.exec_times.append(job.exec_time_us)
        e.total_latencies.append(job.total_latency_us)
        e.queue_times.append(job.queue_us)

    return entities


def percentile(data: list[float], pct: int) -> float:
    """Compute percentile from unsorted data."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(len(s) * pct / 100)
    if idx >= len(s):
        idx = len(s) - 1
    return s[idx]


def _print_latency_block(name: str, data: list[float], show_percentiles: bool = True):
    """Print min/avg/max and optionally p50/p95/p99 for a latency distribution."""
    if not data:
        return
    avg = sum(data) / len(data)
    print(f"    {name} (us):")
    print(f"      min={min(data):.0f}  avg={avg:.0f}  max={max(data):.0f}")
    if show_percentiles:
        print(f"      p50={percentile(data, 50):.0f}  "
              f"p95={percentile(data, 95):.0f}  "
              f"p99={percentile(data, 99):.0f}")


def print_entity_stats(label: str, entities: dict[int, EntityStats]):
    """Print formatted statistics for all entities."""
    total_exec = sum(e.total_exec_us for e in entities.values())

    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")

    for cid in sorted(entities.keys()):
        e = entities[cid]
        share = (e.total_exec_us / total_exec * 100) if total_exec else 0

        print(f"\n  Entity {cid} (ring={e.ring}):")
        print(f"    Jobs: {e.job_count}   GPU share: {share:.1f}%")

        _print_latency_block("Queue latency", e.queue_latencies)
        _print_latency_block("Exec time", e.exec_times, show_percentiles=False)
        _print_latency_block("Total latency", e.total_latencies)


def write_csv(outdir: str, label: str, entities: dict[int, EntityStats]):
    """Write per-entity latency data to CSV files."""
    os.makedirs(outdir, exist_ok=True)

    for cid, e in entities.items():
        fname = os.path.join(outdir, f"{label}_entity_{cid}.csv")
        with open(fname, "w") as f:
            f.write("job_idx,queue_latency_us,exec_time_us,total_latency_us,"
                    "queue_ms\n")
            t0 = min(e.queue_times) if e.queue_times else 0
            for i, (ql, et, tl, qt) in enumerate(zip(
                    e.queue_latencies, e.exec_times, e.total_latencies,
                    e.queue_times)):
                f.write(f"{i},{ql:.1f},{et:.1f},{tl:.1f},{(qt - t0) / 1000:.1f}\n")
        print(f"  Wrote {fname}")


def write_summary_csv(outdir: str, all_stats: dict):
    """Write a comparison summary CSV across policies."""
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, "summary.csv")

    with open(fname, "w") as f:
        f.write("policy,entity,jobs,gpu_share_pct,"
                "queue_lat_p50,queue_lat_p95,queue_lat_p99,"
                "total_lat_p50,total_lat_p95,total_lat_p99\n")

        for label, entities in all_stats.items():
            total_exec = sum(e.total_exec_us for e in entities.values())
            for cid in sorted(entities.keys()):
                e = entities[cid]
                share = (e.total_exec_us / total_exec * 100) if total_exec else 0
                f.write(f"{label},{cid},{e.job_count},{share:.1f},"
                        f"{percentile(e.queue_latencies, 50):.0f},"
                        f"{percentile(e.queue_latencies, 95):.0f},"
                        f"{percentile(e.queue_latencies, 99):.0f},"
                        f"{percentile(e.total_latencies, 50):.0f},"
                        f"{percentile(e.total_latencies, 95):.0f},"
                        f"{percentile(e.total_latencies, 99):.0f}\n")

    print(f"\nWrote summary to {fname}")


# ── Plotting ──────────────────────────────────────────────────────────────
#
# All plot functions require matplotlib + numpy (imported lazily so
# analysis-only usage works without them).

def _init_plotting():
    """Lazy import and configure matplotlib. Returns (plt, np, ticker)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Noto Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'normal',
        'legend.fontsize': 8,
        'legend.frameon': False,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'svg.fonttype': 'path',
        'figure.figsize': (10, 4),
    })
    return plt, np, ticker


POLICY_COLORS = {
    'fifo': '#d62728',
    'fair': '#1f77b4',
    'deadline': '#2ca02c',
}
POLICY_LABELS = {
    'fifo': 'FIFO',
    'fair': 'Fair',
    'deadline': 'Deadline',
}
CLIENT_STYLES = {0: '-', 1: '--', 2: ':', 3: '-.'}


def _load_bench_csv(path):
    """Load benchmark CSV. Returns list of (submit_ns, complete_ns, elapsed_ns)."""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 4:
                rows.append((int(row[1]), int(row[2]), int(row[3])))
    return rows


def _find_bench_csvs(scenario_dir, policy, labels):
    """Find benchmark CSV files for a policy and list of client labels."""
    result = {}
    for label in labels:
        path = os.path.join(scenario_dir, f'{policy}_{label}.csv')
        if os.path.exists(path):
            result[label] = _load_bench_csv(path)
    return result


def _load_trace_csv(path):
    """Load trace-generated CSV (job_idx, queue_latency_us, exec_time_us, total_latency_us)."""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4:
                t = float(row[4]) if len(row) >= 5 else None
                rows.append((int(row[0]), float(row[1]), float(row[2]),
                             float(row[3]), t))
    return rows


def _find_trace_entities(scenario_dir, policy):
    """
    Find all benchmark entity trace CSVs for a given policy.
    Filters out non-benchmark entities (SDMA copy engines, amdgpu internals).
    Returns dict of {entity_id: rows}.
    """
    import glob as globmod

    csv_dir = os.path.join(scenario_dir, 'csv')
    if not os.path.isdir(csv_dir):
        return {}

    SDMA_ID_THRESHOLD = 10000  # SDMA copy engines use high entity IDs
    AMDGPU_INTERNAL_ENTITIES = {11, 27}  # amdgpu driver-internal entities
    pattern = os.path.join(csv_dir, f'{policy}_entity_*.csv')
    entities = {}

    for path in globmod.glob(pattern):
        fname = os.path.basename(path)
        eid_str = fname.replace(f'{policy}_entity_', '').replace('.csv', '')
        try:
            eid = int(eid_str)
        except ValueError:
            continue
        if eid > SDMA_ID_THRESHOLD or eid in AMDGPU_INTERNAL_ENTITIES:
            continue
        rows = _load_trace_csv(path)
        if rows:
            entities[eid] = rows

    return entities


def _find_light_heavy_entities(scenario_dir, policy):
    """
    Identify the light (rate-limited) and heavy (saturating) entities in S3.

    Combined heuristic:
    1. If avg exec times differ by >1.5x → light has shorter exec (S3b).
    2. Otherwise → light has fewer completed jobs (S3a, same exec times).

    Returns (light_id, light_rows, heavy_id, heavy_rows).
    """
    entities = _find_trace_entities(scenario_dir, policy)
    if not entities:
        return None, None, None, None

    # Need at least 2 benchmark entities to distinguish light vs heavy
    if len(entities) < 2:
        eid, rows = next(iter(entities.items()))
        return eid, rows, None, None

    # Pick the two entities with the most jobs (skip tiny internal ones)
    by_count = sorted(entities.items(), key=lambda x: len(x[1]), reverse=True)
    top_two = by_count[:2]

    # Combined heuristic to distinguish light (rate-limited) from heavy (saturating):
    # 1. If avg exec times differ by >1.5x, heavy has longer exec → light has shorter.
    #    Works for S3b (heavy=10ms, light=1ms) even when heavy has fewer total jobs.
    # 2. Otherwise (S3a: both ~1ms), fall back to job count: light has fewer jobs
    #    because it's rate-limited while heavy saturates.
    def avg_exec(rows):
        execs = [r[2] for r in rows if r[2] > 0]  # exec_time_us is index 2
        return sum(execs) / len(execs) if execs else 0

    avg0 = avg_exec(top_two[0][1])
    avg1 = avg_exec(top_two[1][1])

    if avg0 > 0 and avg1 > 0 and max(avg0, avg1) / min(avg0, avg1) > 1.5:
        # Exec times differ enough — light entity has shorter jobs
        if avg0 <= avg1:
            light_id, light_rows = top_two[0]
            heavy_id, heavy_rows = top_two[1]
        else:
            light_id, light_rows = top_two[1]
            heavy_id, heavy_rows = top_two[0]
    else:
        # Similar exec times — light entity has fewer completed jobs
        if len(top_two[0][1]) <= len(top_two[1][1]):
            light_id, light_rows = top_two[0]
            heavy_id, heavy_rows = top_two[1]
        else:
            light_id, light_rows = top_two[1]
            heavy_id, heavy_rows = top_two[0]

    return light_id, light_rows, heavy_id, heavy_rows


# ── Individual figure functions ───────────────────────────────────────────

def _plot_fairness(results_dir, output_dir, plt, np):
    """Fig 1: S1 cumulative completions, two equal entities."""
    scenario = os.path.join(results_dir, 'S1_fairness')
    if not os.path.isdir(scenario):
        print('  Skipping S1: not found')
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    policies = ['fifo', 'fair', 'deadline']
    labels = ['client_a', 'client_b']
    label_names = ['Client A', 'Client B']

    for ax, policy in zip(axes, policies):
        data = _find_bench_csvs(scenario, policy, labels)
        all_submits = [r[0] for l in labels if l in data for r in data[l]]
        if not all_submits:
            continue
        t0 = min(all_submits)

        for i, label in enumerate(labels):
            if label not in data or not data[label]:
                continue
            rows = data[label]
            times_s = sorted([(r[1] - t0) / 1e9 for r in rows])
            cumulative = np.arange(1, len(times_s) + 1)
            ax.plot(times_s, cumulative, color=f'C{i}',
                    linestyle=CLIENT_STYLES[i], linewidth=1.5,
                    label=label_names[i])

        ax.set_title(POLICY_LABELS[policy], color=POLICY_COLORS[policy],
                     fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Cumulative completed jobs')
    plt.tight_layout()
    out = os.path.join(output_dir, 'fig1_fairness_s1.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def _plot_gpu_share(results_dir, output_dir, plt, np):
    """Fig 2: S2 GPU share bar chart with ratio annotations."""
    scenario = os.path.join(results_dir, 'S2_proportional')
    if not os.path.isdir(scenario):
        print('  Skipping S2 share: not found')
        return

    policies = ['fifo', 'fair', 'deadline']
    med_counts = []
    low_counts = []

    for policy in policies:
        data = _find_bench_csvs(scenario, policy, ['medium', 'low'])
        if data:
            med_counts.append(len(data.get('medium', [])))
            low_counts.append(len(data.get('low', [])))
        else:
            med_counts.append(0)
            low_counts.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(policies))
    width = 0.3

    ax.bar(x - width/2, med_counts, width, label='Medium (w=64)',
           color='#2196F3', alpha=0.8)
    ax.bar(x + width/2, low_counts, width, label='Low (w=8)',
           color='#FF9800', alpha=0.8)

    ax.set_ylabel('Completed jobs')
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for i, policy in enumerate(policies):
        if low_counts[i] > 0:
            ratio = med_counts[i] / low_counts[i]
            ax.text(i, max(med_counts[i], low_counts[i]) + 50,
                    f'{ratio:.1f}:1', ha='center', fontsize=11,
                    fontweight='bold', color=POLICY_COLORS[policy])
        else:
            ax.text(i, med_counts[i] + 50, 'starvation', ha='center',
                    fontsize=10, fontweight='bold', color=POLICY_COLORS[policy])

    ax.axhline(y=0, color='black', linewidth=0.5)

    out = os.path.join(output_dir, 'fig2_priority_detail_s2.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def _plot_interactive(results_dir, output_dir, plt, np):
    """Fig 3: S3 latency CDF for light entity."""
    scenarios = [
        ('S3a_interactive', 'Heavy 1ms saturating + Light 1ms @100Hz'),
        ('S3b_interactive_long', 'Heavy 10ms saturating + Light 1ms @100Hz'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    policies = ['fifo', 'fair', 'deadline']

    for ax, (scenario_name, subtitle) in zip(axes, scenarios):
        if not os.path.isdir(os.path.join(results_dir, scenario_name)):
            print(f'  Skipping {scenario_name}: not found')
            continue

        for policy in policies:
            # Kernel-side queue-to-done latency (matches the results table).
            lats = _scenario_trace_latencies(results_dir, scenario_name, policy)
            if not lats:
                continue
            latencies_ms = sorted(l / 1000 for l in lats)
            cdf = np.arange(1, len(latencies_ms) + 1) / len(latencies_ms)
            ax.plot(latencies_ms, cdf, color=POLICY_COLORS[policy],
                    linewidth=2,
                    label=f'{POLICY_LABELS[policy]} (n={len(lats):,})')

        ax.set_xlabel('Job latency (ms)')
        ax.set_title(subtitle, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
        ax.text(10.5, 0.5, '10ms\nbudget', fontsize=8, color='gray', alpha=0.7)

    axes[0].set_ylabel('CDF')
    plt.tight_layout()
    out = os.path.join(output_dir, 'fig3_latency_s3.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def _plot_latency_timeline(results_dir, output_dir, plt, np):
    """Fig 4: S3 per-job latency over time for light entity (trace data)."""
    scenarios = [
        ('S3a_interactive', 'S3a: Heavy 1ms + Light @100Hz'),
        ('S3b_interactive_long', 'S3b: Heavy 10ms + Light @100Hz'),
    ]
    policies = ['fifo', 'fair', 'deadline']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, (scenario_name, row_label) in enumerate(scenarios):
        scenario = os.path.join(results_dir, scenario_name)
        if not os.path.isdir(scenario):
            print(f'  Skipping {scenario_name}: not found')
            continue

        y_max_row = 0
        for col, policy in enumerate(policies):
            ax = axes[row][col]
            eid, trace_rows, _, _ = _find_light_heavy_entities(scenario, policy)

            if trace_rows is None or len(trace_rows) < 10:
                n = len(trace_rows) if trace_rows else 0
                ax.text(0.5, 0.5, f'Light entity starved\n({n} jobs)',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, color='red', fontweight='bold')
                if row == 0:
                    ax.set_title(POLICY_LABELS[policy],
                                 color=POLICY_COLORS[policy],
                                 fontweight='bold', fontsize=13)
                if col == 0:
                    ax.set_ylabel(f'{row_label}\nTotal latency (ms)')
                continue

            # Real queue time from the trace; fall back to job index for
            # older CSVs that predate the queue_ms column.
            if trace_rows[0][4] is not None:
                times_s = np.array([r[4] / 1000.0 for r in trace_rows])
            else:
                times_s = np.array([r[0] * 0.01 for r in trace_rows])
            total_lat_ms = np.array([r[3] / 1000.0 for r in trace_rows])
            queue_lat_ms = np.array([r[1] / 1000.0 for r in trace_rows])

            ax.scatter(times_s, total_lat_ms, s=4, alpha=0.3,
                       color=POLICY_COLORS[policy], rasterized=True,
                       label='Total latency')

            window = max(10, min(50, len(total_lat_ms) // 10))
            if len(total_lat_ms) >= window:
                med_times = []
                med_vals = []
                for start in range(0, len(total_lat_ms) - window + 1,
                                   max(1, window // 4)):
                    end = start + window
                    med_times.append(np.mean(times_s[start:end]))
                    med_vals.append(np.median(total_lat_ms[start:end]))
                ax.plot(med_times, med_vals, color='black', linewidth=1.5,
                        alpha=0.8, label='Rolling median')

            ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5)

            n = len(trace_rows)
            p50 = np.median(total_lat_ms)
            p99 = np.percentile(total_lat_ms, 99)
            q_p50 = np.median(queue_lat_ms)
            stats_text = (f'n={n:,}  entity={eid}\n'
                          f'p50={p50:.2f}ms  p99={p99:.2f}ms\n'
                          f'queue p50={q_p50:.2f}ms')
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.8, edgecolor='gray'))

            y_max_row = max(y_max_row, np.percentile(total_lat_ms, 99.5))

            if row == 0:
                ax.set_title(POLICY_LABELS[policy],
                             color=POLICY_COLORS[policy],
                             fontweight='bold', fontsize=13)
            if col == 0:
                ax.set_ylabel(f'{row_label}\nTotal latency (ms)')
            if row == 1:
                ax.set_xlabel('Time (s)')

            ax.grid(True, alpha=0.2)
            ax.set_ylim(bottom=0)
            ax.set_xlim(0, 20)

        if y_max_row > 0:
            for col in range(3):
                axes[row][col].set_ylim(0, y_max_row * 1.1)

    plt.tight_layout()
    out = os.path.join(output_dir, 'fig4_latency_timeline_s3.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def _plot_scaling_detail(results_dir, output_dir, plt, np):
    """Fig 5: S4 cumulative completions + per-entity share bars."""
    scenario = os.path.join(results_dir, 'S4_scaling')
    if not os.path.isdir(scenario):
        print('  Skipping S4 detail: not found')
        return

    policies = ['fifo', 'fair', 'deadline']
    clients = ['client_a', 'client_b', 'client_c', 'client_d']
    client_labels = ['A', 'B', 'C', 'D']
    client_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             gridspec_kw={'height_ratios': [1.2, 1]})

    for col, policy in enumerate(policies):
        ax_cum = axes[0][col]
        ax_bar = axes[1][col]

        data = _find_bench_csvs(scenario, policy, clients)
        all_rows = []
        for c in clients:
            if c in data:
                all_rows.extend(data[c])
        if not all_rows:
            continue
        t0 = min(r[0] for r in all_rows)

        counts = []
        for i, client in enumerate(clients):
            if client not in data or not data[client]:
                counts.append(0)
                continue
            rows = data[client]
            times_s = sorted([(r[1] - t0) / 1e9 for r in rows])
            cumulative = np.arange(1, len(times_s) + 1)
            ax_cum.plot(times_s, cumulative, color=client_colors[i],
                        linestyle=CLIENT_STYLES[i], linewidth=1.5,
                        label=f'Client {client_labels[i]}')
            counts.append(len(rows))

        ax_cum.set_title(POLICY_LABELS[policy], color=POLICY_COLORS[policy],
                         fontweight='bold', fontsize=13)
        ax_cum.grid(True, alpha=0.3)
        if col == 0:
            ax_cum.set_ylabel('Cumulative jobs')
        ax_cum.legend(loc='upper left', fontsize=8)

        x = np.arange(len(clients))
        total = sum(counts)
        shares = [c / total * 100 if total > 0 else 0 for c in counts]

        ax_bar.bar(x, shares, color=client_colors, alpha=0.8,
                   edgecolor='white', linewidth=0.5)
        ax_bar.axhline(y=25, color='gray', linestyle='--', alpha=0.7,
                       linewidth=1, label='Ideal 25%')

        for j, (share, count) in enumerate(zip(shares, counts)):
            ax_bar.text(j, share + 1, f'{share:.1f}%\n({count:,})',
                        ha='center', va='bottom', fontsize=8)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f'Client {l}' for l in client_labels],
                               fontsize=9)
        ax_bar.set_ylim(0, max(shares) * 1.3 if max(shares) > 0 else 100)
        ax_bar.grid(True, alpha=0.2, axis='y')
        if col == 0:
            ax_bar.set_ylabel('GPU share (%)')

        if total > 0:
            cv = np.std(counts) / np.mean(counts) * 100
            ax_bar.text(0.95, 0.92, f'CV = {cv:.1f}%',
                        transform=ax_bar.transAxes, ha='right', va='top',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='lightyellow',
                                  edgecolor='orange', alpha=0.9))

    plt.tight_layout()
    out = os.path.join(output_dir, 'fig5_scaling_detail_s4.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def _jains_fairness(shares):
    """Jain's fairness index: 1.0 = perfectly fair, 1/n = maximally unfair."""
    if not shares or all(s == 0 for s in shares):
        return 0.0
    n = len(shares)
    return sum(shares) ** 2 / (n * sum(s ** 2 for s in shares))


def _scenario_shares(results_dir, scenario_name, policy, labels):
    """Get per-client job counts for a scenario/policy."""
    scenario = os.path.join(results_dir, scenario_name)
    data = _find_bench_csvs(scenario, policy, labels)
    return [len(data.get(l, [])) for l in labels]


def _scenario_trace_latencies(results_dir, scenario_name, policy):
    """Get light entity's total_latency_us from trace CSV (kernel-side timing).

    For rate-limited clients, bench CSV elapsed_ns measures ring buffer cycle
    time, not scheduling latency. Trace data gives the real queue-to-done time.
    Returns list of total_latency_us floats, or empty list if unavailable.
    """
    scenario = os.path.join(results_dir, scenario_name)
    _, light_rows, _, _ = _find_light_heavy_entities(scenario, policy)
    if not light_rows:
        return []
    return [r[3] for r in light_rows]  # total_latency_us


def _windowed_jain_min(results_dir, scenario_name, policy, labels,
                       window_s=1.0):
    """Minimum Jain's fairness index over sliding time windows.

    Total Jain's index can be 1.0 even when one entity monopolizes for long
    stretches (zig-zag). Windowed analysis captures temporal unfairness.
    """
    scenario = os.path.join(results_dir, scenario_name)
    data = _find_bench_csvs(scenario, policy, labels)

    # Collect completion times per label
    per_label = []
    for label in labels:
        if label not in data:
            per_label.append([])
        else:
            per_label.append(sorted([r[1] / 1e9 for r in data[label]]))

    all_times = sorted(t for times in per_label for t in times)
    if len(all_times) < 2:
        return 1.0

    t_start = all_times[0]
    t_end = all_times[-1]
    step = window_s / 4
    min_jain = 1.0
    t = t_start

    while t + window_s <= t_end:
        w_end = t + window_s
        counts = []
        for times in per_label:
            # Binary search would be faster but n is small enough
            c = sum(1 for x in times if t <= x < w_end)
            counts.append(c)
        total = sum(counts)
        if total > 0:
            n = len(counts)
            jain = total ** 2 / (n * sum(c ** 2 for c in counts))
            min_jain = min(min_jain, jain)
        t += step

    return min_jain


def generate_summary_table(results_dir, output_dir):
    """Generate a summary table: text + LaTeX."""
    policies = ['fifo', 'fair', 'deadline']
    policy_hdr = {'fifo': 'FIFO', 'fair': 'Fair', 'deadline': 'Deadline'}

    rows = []

    # S1: Windowed Jain's fairness index (captures temporal unfairness)
    s1_labels = ['client_a', 'client_b']
    s1_row = ['S1: Jain index (1s window min)']
    for p in policies:
        jfi = _windowed_jain_min(results_dir, 'S1_fairness', p, s1_labels)
        s1_row.append(f'{jfi:.4f}')
    rows.append(s1_row)

    # S1: max/min share ratio
    s1_ratio_row = ['S1: max/min job ratio']
    for p in policies:
        shares = _scenario_shares(results_dir, 'S1_fairness', p, s1_labels)
        if min(shares) > 0:
            s1_ratio_row.append(f'{max(shares)/min(shares):.2f}')
        else:
            s1_ratio_row.append('$\\infty$')
    rows.append(s1_ratio_row)

    # S2: medium/low ratio (target: 8.0)
    s2_row = ['S2: med/low ratio (target 8:1)']
    for p in policies:
        shares = _scenario_shares(results_dir, 'S2_proportional', p,
                                  ['medium', 'low'])
        if shares[1] > 0:
            s2_row.append(f'{shares[0]/shares[1]:.1f}:1')
        else:
            s2_row.append('starvation')
    rows.append(s2_row)

    # S3a: light entity latency from trace data (kernel-side timing)
    for pct_label, pct in [('p50', 50), ('p99', 99)]:
        row = [f'S3a: light latency {pct_label} (ms)']
        for p in policies:
            lats = _scenario_trace_latencies(results_dir,
                                             'S3a_interactive', p)
            if lats:
                val = percentile(lats, pct) / 1000  # us -> ms
                row.append(f'{val:.2f}')
            else:
                row.append('--')
        rows.append(row)

    # S3b: light entity latency from trace data
    for pct_label, pct in [('p50', 50), ('p99', 99)]:
        row = [f'S3b: light latency {pct_label} (ms)']
        for p in policies:
            lats = _scenario_trace_latencies(results_dir,
                                             'S3b_interactive_long', p)
            if lats:
                val = percentile(lats, pct) / 1000  # us -> ms
                row.append(f'{val:.2f}')
            else:
                row.append('--')
        rows.append(row)

    # S3b: light entity job count (starvation indicator)
    s3b_count_row = ['S3b: light jobs completed']
    for p in policies:
        lats = _scenario_trace_latencies(results_dir,
                                         'S3b_interactive_long', p)
        s3b_count_row.append(str(len(lats)) if lats else '--')
    rows.append(s3b_count_row)

    # S4: Windowed Jain's fairness + CV
    s4_labels = ['client_a', 'client_b', 'client_c', 'client_d']
    s4_jfi_row = ['S4: Jain index (1s window min)']
    s4_cv_row = ['S4: GPU share CV (%)']
    for p in policies:
        jfi = _windowed_jain_min(results_dir, 'S4_scaling', p, s4_labels)
        s4_jfi_row.append(f'{jfi:.4f}')
        shares = _scenario_shares(results_dir, 'S4_scaling', p, s4_labels)
        mean_s = sum(shares) / len(shares) if shares else 0
        if mean_s > 0:
            import math
            std_s = math.sqrt(sum((s - mean_s)**2 for s in shares) / len(shares))
            s4_cv_row.append(f'{std_s / mean_s * 100:.1f}')
        else:
            s4_cv_row.append('--')
    rows.append(s4_jfi_row)
    rows.append(s4_cv_row)

    # Print text table
    col_widths = [max(len(r[i]) for r in rows) for i in range(4)]
    col_widths[0] = max(col_widths[0], 30)
    for i in range(1, 4):
        col_widths[i] = max(col_widths[i], 10)

    hdr = f'{"Metric":<{col_widths[0]}}  ' + '  '.join(
        f'{policy_hdr[p]:>{col_widths[i+1]}}' for i, p in enumerate(policies))
    sep = '-' * len(hdr)

    print(f'\n{sep}')
    print(f'  POLICY COMPARISON SUMMARY')
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        line = f'{row[0]:<{col_widths[0]}}  ' + '  '.join(
            f'{row[i+1]:>{col_widths[i+1]}}' for i in range(3))
        print(line)
    print(sep)

    # Write LaTeX table
    os.makedirs(output_dir, exist_ok=True)
    tex_path = os.path.join(output_dir, 'summary_table.tex')
    with open(tex_path, 'w') as f:
        f.write('% Auto-generated by analyze.py --plot\n')
        f.write('\\begin{table}[htbp]\n')
        f.write('\\centering\n')
        f.write('\\caption{Policy comparison across benchmark scenarios.}\n')
        f.write('\\label{tab:policy-comparison}\n')
        f.write('\\begin{tabular}{l r r r}\n')
        f.write('\\toprule\n')
        f.write('Metric & FIFO & Fair & Deadline \\\\\n')
        f.write('\\midrule\n')
        for row in rows:
            # Escape underscores and handle special values
            metric = row[0].replace('_', '\\_').replace('%', '\\%')
            vals = ' & '.join(row[1:])
            f.write(f'{metric} & {vals} \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')

    print(f'\n  Saved {tex_path}')
    return rows


def run_plots(results_dir, output_dir):
    """Generate all figures from a benchmark results directory."""
    plt, np, ticker = _init_plotting()
    os.makedirs(output_dir, exist_ok=True)

    print(f'Generating plots from {results_dir}')
    print(f'Output: {output_dir}\n')

    _plot_fairness(results_dir, output_dir, plt, np)
    _plot_gpu_share(results_dir, output_dir, plt, np)
    _plot_interactive(results_dir, output_dir, plt, np)
    _plot_latency_timeline(results_dir, output_dir, plt, np)
    _plot_scaling_detail(results_dir, output_dir, plt, np)
    generate_summary_table(results_dir, output_dir)

    print('\nDone. Figures:')
    print('  fig1 — S1 Fairness:          cumulative completions (equal 50%+50%)')
    print('  fig2 — S2 Priority detail:   GPU share bar chart with ratio')
    print('  fig3 — S3 Latency:           CDF (short and long heavy jobs)')
    print('  fig4 — S3 Latency timeline:  per-job latency over time')
    print('  fig5 — S4 Scaling detail:    cumulative + per-entity share bars')
    print('  table — summary_table.tex:   LaTeX-ready policy comparison')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze DRM scheduler ftrace output and generate plots")
    parser.add_argument("traces", nargs="*",
                        help="Trace files (named like trace_fifo.txt)")
    parser.add_argument("--csv", metavar="DIR",
                        help="Write CSV output to directory")
    parser.add_argument("--plot", metavar="RESULTS_DIR",
                        help="Generate plots from results directory")
    parser.add_argument("--plot-output", metavar="DIR", default=None,
                        help="Output directory for plots (default: same as RESULTS_DIR)")
    parser.add_argument("--table", metavar="RESULTS_DIR",
                        help="Generate summary table only (text + LaTeX)")
    args = parser.parse_args()

    if args.table:
        output_dir = args.plot_output or args.table
        generate_summary_table(args.table, output_dir)
        return

    if args.plot:
        output_dir = args.plot_output or args.plot
        run_plots(args.plot, output_dir)
        return

    if not args.traces:
        parser.error("either --plot or trace files required")

    all_stats = {}

    for filepath in args.traces:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        label = basename.replace("trace_", "")

        print(f"Parsing {filepath} (label={label})...")
        jobs = parse_trace_file(filepath)
        complete = sum(1 for j in jobs.values()
                       if j.queue_us and j.run_us and j.done_us)
        print(f"  {len(jobs)} jobs found, {complete} complete")

        entities = compute_entity_stats(jobs)
        all_stats[label] = entities

        print_entity_stats(label, entities)

        if args.csv:
            write_csv(args.csv, label, entities)

    if args.csv and len(all_stats) > 1:
        write_summary_csv(args.csv, all_stats)


if __name__ == "__main__":
    main()
