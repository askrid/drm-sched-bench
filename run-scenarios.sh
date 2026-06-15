#!/bin/bash
# Runs all scenarios under all three policies (FIFO, fair, deadline),
# collects ftrace data, and produces per-scenario analysis.
#
# Usage:
#   sudo bash run-scenarios.sh [--duration <ms>] [--output-dir <dir>] [--device <idx>]
#
# Checklist before running:
#   1. uname -r
#   2. /sys/module/gpu_sched/parameters/sched_policy
#   3. VK_DRIVER_FILES points to radv

set -eo pipefail

# ── Defaults ────────────────────────────────────────────────────────────
DURATION_MS=20000
OUTDIR="$(dirname "$0")/results/$(date +%Y%m%d-%H%M%S)"
VK_DEVICE_IDX=0
BENCH="$(dirname "$0")/vkload"
TRACEDIR=/sys/kernel/tracing
POLICY_PARAM=/sys/module/gpu_sched/parameters/sched_policy

# ── Parse args ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)   DURATION_MS="$2"; shift 2 ;;
        --output-dir) OUTDIR="$2"; shift 2 ;;
        --device)     VK_DEVICE_IDX="$2"; shift 2 ;;
        --help)
            echo "Usage: sudo bash $0 [--duration <ms>] [--output-dir <dir>] [--device <idx>]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Detect AMD Vulkan ICD ──────────────────────────────────────────────
RADV_ICD=""
for f in /usr/share/vulkan/icd.d/radeon_icd*.json; do
    [ -f "$f" ] && RADV_ICD="$f" && break
done
if [ -z "$RADV_ICD" ]; then
    echo "ERROR: radv ICD not found. Install vulkan-radeon:"
    echo "  sudo pacman -S vulkan-radeon"
    exit 1
fi
export VK_DRIVER_FILES="$RADV_ICD"

# ── Preflight checks ───────────────────────────────────────────────────
echo "=== DRM Scheduler Benchmark Suite ==="
echo ""

# Check root
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Must run as root (need ftrace + sched_policy access)"
    exit 1
fi

# Check benchmark binary
if [ ! -x "$BENCH" ]; then
    echo "ERROR: $BENCH not found. Build it first:"
    echo "  make -C $(dirname "$0")"
    exit 1
fi

# Check policy param
if [ ! -f "$POLICY_PARAM" ]; then
    echo "ERROR: $POLICY_PARAM not found."
    echo "Are you booted into the thesis kernel?"
    exit 1
fi

# Check tracefs
if [ ! -d "$TRACEDIR" ]; then
    echo "ERROR: tracefs not mounted"
    exit 1
fi

# Check AMD GPU is active
AMD_CARD=""
for card in /sys/class/drm/card*; do
    [ -d "$card/device" ] || continue
    driver=$(basename "$(readlink "$card/device/driver" 2>/dev/null)" 2>/dev/null)
    if [ "$driver" = "amdgpu" ]; then
        AMD_CARD=$(basename "$card")
        break
    fi
done
if [ -z "$AMD_CARD" ]; then
    echo "ERROR: No amdgpu card found."
    echo "Check that nomodeset is removed from kernel cmdline."
    exit 1
fi

# Verify Vulkan sees AMD
VK_CHECK=$("$BENCH" --device "$VK_DEVICE_IDX" --calibrate-only --job-us 100 2>&1 || true)
if ! echo "$VK_CHECK" | grep -qi "amd\|radeon\|raphael"; then
    echo "WARNING: Vulkan device $VK_DEVICE_IDX may not be AMD:"
    echo "  $VK_CHECK"
    echo ""
    echo "Try --device 1 or check VK_DRIVER_FILES"
    echo "Continuing anyway..."
fi

echo "AMD GPU:       $AMD_CARD"
echo "Vulkan ICD:    $RADV_ICD"
echo "Duration:      ${DURATION_MS}ms per scenario"
echo "Output:        $OUTDIR"
echo ""

mkdir -p "$OUTDIR"

# ── Tracing helpers ─────────────────────────────────────────────────────
trace_start() {
    echo 0 > "$TRACEDIR/tracing_on"
    echo > "$TRACEDIR/trace"
    echo 65536 > "$TRACEDIR/buffer_size_kb"
    echo 0 > "$TRACEDIR/events/enable"
    echo 1 > "$TRACEDIR/events/gpu_scheduler/drm_sched_job_queue/enable"
    echo 1 > "$TRACEDIR/events/gpu_scheduler/drm_sched_job_run/enable"
    echo 1 > "$TRACEDIR/events/gpu_scheduler/drm_sched_job_done/enable"
    [ -f "$TRACEDIR/events/gpu_scheduler/drm_sched_deadline_pick/enable" ] && \
        echo 1 > "$TRACEDIR/events/gpu_scheduler/drm_sched_deadline_pick/enable"
    echo 1 > "$TRACEDIR/tracing_on"
}

trace_stop() {
    local outfile="$1"
    echo 0 > "$TRACEDIR/tracing_on"
    cat "$TRACEDIR/trace" > "$outfile"
    echo 0 > "$TRACEDIR/events/gpu_scheduler/enable"
    local lines
    lines=$(wc -l < "$outfile")
    echo "  trace: $lines lines → $outfile"
}

# ── Scenario runner ─────────────────────────────────────────────────────
# run_scenario <policy_name> <policy_num> <scenario_name> <client_specs...>
# Each client_spec: "job_us:rate_hz:queue_depth:label[:priority[:burst]]"
# priority is optional: low, medium, high, realtime (use '-' to skip)
# burst is optional: submit N jobs, wait all, repeat
run_scenario() {
    local policy_name="$1" policy_num="$2" scenario="$3"
    shift 3

    local scenario_dir="$OUTDIR/$scenario"
    mkdir -p "$scenario_dir"

    local num_clients=$#
    echo "  [$policy_name] $scenario ($num_clients clients)"

    # Set policy and verify
    echo "$policy_num" > "$POLICY_PARAM"
    local actual
    actual=$(cat "$POLICY_PARAM")
    if [ "$actual" != "$policy_num" ]; then
        echo "  ERROR: policy set to $policy_num but read back $actual"
        return 1
    fi

    # Prepare sync barrier directory
    local sync_dir="$scenario_dir/.sync_${policy_name}"
    rm -rf "$sync_dir"
    mkdir -p "$sync_dir"

    # Start tracing
    trace_start

    # Launch all clients
    local pids=()
    for spec in "$@"; do
        IFS=: read -r job_us rate_hz qdepth label prio burst <<< "$spec"
        local csv="$scenario_dir/${policy_name}_${label}.csv"
        local extra_args=""
        if [ -n "$prio" ] && [ "$prio" != "-" ]; then
            extra_args="--priority $prio"
        fi
        if [ -n "$burst" ] && [ "$burst" -gt 0 ] 2>/dev/null; then
            extra_args="$extra_args --burst $burst"
        fi
        "$BENCH" --device "$VK_DEVICE_IDX" \
               --job-us "$job_us" \
               --rate "${rate_hz:-0}" \
               --queue-depth "${qdepth:-4}" \
               --duration "$DURATION_MS" \
               --output "$csv" \
               --sync-dir "$sync_dir" \
               --sync-count "$num_clients" \
               $extra_args &
        pids+=($!)
    done

    # Wait for all clients
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: client PID $pid exited with error"
            failed=$((failed + 1))
        fi
    done

    # Stop tracing
    trace_stop "$scenario_dir/trace_${policy_name}.txt"

    # Clean up sync dir
    rm -rf "$sync_dir"

    if [ "$failed" -gt 0 ]; then
        echo "  WARNING: $failed client(s) failed in $scenario/$policy_name"
    fi
}

# ── Scenario definitions ───────────────────────────────────────────────
# Spec: "job_us:rate_hz:queue_depth:label[:priority[:burst]]" ('-' skips prio).
# rate=0 saturates (resubmit as fast as the GPU drains).

POLICIES=("fifo:0" "fair:1" "deadline:2")

# Calibrate once to verify GPU duration accuracy
echo "Calibrating GPU job durations..."
"$BENCH" --device "$VK_DEVICE_IDX" --job-us 1000 --calibrate-only 2>&1 | tee "$OUTDIR/calibration.txt"
echo ""

for policy_entry in "${POLICIES[@]}"; do
    IFS=: read -r pname pnum <<< "$policy_entry"
    echo "=== Policy: $pname (id=$pnum) ==="

    # S1 Fairness: two equal 5ms entities, deep queue (48) to expose FIFO bursts.
    run_scenario "$pname" "$pnum" "S1_fairness" \
        "5000:0:48:client_a" \
        "5000:0:48:client_b"

    # S2 Proportional: medium (w=64) vs low (w=8), same 5ms job. Target 8:1.
    run_scenario "$pname" "$pnum" "S2_proportional" \
        "5000:0:4:medium:medium" \
        "5000:0:4:low:low"

    # S3a Interactive: heavy 1ms saturating + light 1ms @100Hz. Undersubscribed.
    run_scenario "$pname" "$pnum" "S3a_interactive" \
        "1000:0:16:heavy" \
        "1000:100:4:light"

    # S3b Interactive: heavy 10ms saturating + light 1ms @100Hz. Oversubscribed.
    run_scenario "$pname" "$pnum" "S3b_interactive_long" \
        "10000:0:16:heavy" \
        "1000:100:4:light"

    # S4 Scaling: four equal 1ms entities, deep queue (48). Fairness at N=4.
    run_scenario "$pname" "$pnum" "S4_scaling" \
        "1000:0:48:client_a" \
        "1000:0:48:client_b" \
        "1000:0:48:client_c" \
        "1000:0:48:client_d"

    echo ""
done

# ── Restore default policy ──────────────────────────────────────────────
echo 1 > "$POLICY_PARAM"  # restore to fair
echo "Restored policy to fair (1)"
echo ""

# ── Analysis ────────────────────────────────────────────────────────────
echo "=== Analyzing results ==="
ANALYZE="$(dirname "$0")/analyze.py"

for scenario_dir in "$OUTDIR"/S*; do
    scenario=$(basename "$scenario_dir")
    echo "--- $scenario ---"

    # Collect trace files for this scenario
    traces=()
    for t in "$scenario_dir"/trace_*.txt; do
        [ -f "$t" ] && traces+=("$t")
    done

    if [ ${#traces[@]} -gt 0 ]; then
        python3 "$ANALYZE" "${traces[@]}" \
            --csv "$scenario_dir/csv" \
            2>&1 | tee "$scenario_dir/analysis.txt"
    fi
    echo ""
done

# ── Per-scenario CSV summary consolidation ──────────────────────────────
echo "=== Generating consolidated summary ==="
SUMMARY="$OUTDIR/summary.csv"
echo "scenario,policy,entity,jobs,gpu_share_pct,queue_lat_p50,queue_lat_p95,queue_lat_p99,total_lat_p50,total_lat_p95,total_lat_p99" > "$SUMMARY"

for scenario_dir in "$OUTDIR"/S*; do
    scenario=$(basename "$scenario_dir")
    csv_file="$scenario_dir/csv/summary.csv"
    if [ -f "$csv_file" ]; then
        # Append with scenario column prepended (skip header)
        tail -n +2 "$csv_file" | while IFS= read -r line; do
            echo "$scenario,$line" >> "$SUMMARY"
        done
    fi
done

echo "Consolidated summary: $SUMMARY"
echo ""

# Hand the results back to the invoking user (this script runs as root).
if [ -n "${SUDO_USER:-}" ]; then
    chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$OUTDIR"
    echo "Ownership of $OUTDIR returned to $SUDO_USER"
fi
echo ""
echo "=== All done ==="
echo "Results in: $OUTDIR"
echo ""
echo "Key files:"
echo "  $OUTDIR/summary.csv           — all metrics in one table"
echo "  $OUTDIR/S*/analysis.txt       — per-scenario text reports"
echo ""
echo "Generate thesis plots:"
echo "  python3 $(dirname "$0")/analyze.py --plot $OUTDIR"
