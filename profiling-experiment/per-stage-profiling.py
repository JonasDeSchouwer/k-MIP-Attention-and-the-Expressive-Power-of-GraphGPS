"""
Per-stage profiling: Full Attention vs k-MIP Attention at N=10^4.5.
Produces a stacked bar chart comparing per-stage computation time.
"""

import gc
import os
import time
import math
import subprocess

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Select the GPU with the most free memory (must be before importing torch)
_result = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    capture_output=True, text=True,
)
if _result.returncode == 0:
    _gpus = []
    for _line in _result.stdout.strip().split("\n"):
        _idx, _free = _line.split(",")
        _gpus.append((int(_idx.strip()), int(_free.strip())))
    _best_gpu = max(_gpus, key=lambda x: x[1])[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_best_gpu)
    print(f"Selected GPU {_best_gpu} (most free memory)")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from methods.symbolic_sparse import symbolic_sparse_nearest_k_keys

# --- Parameters ---
N = 31623  # 10^4.5
k = 10
d_K = 10
val_dim = 10
B = 1
H = 1
NUM_WARMUP = 1
NUM_RUNS = 5
DEVICE = "cuda"
DTYPE = torch.float32

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def sync():
    torch.cuda.synchronize()


# ============================================================
# Full Attention: forward (sync boundaries) + backward (autograd hooks)
# ============================================================

def profile_full_forward(Q, K, V):
    """Profile full attention forward per-stage."""
    sync()

    t0 = time.time()
    attn_scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(d_K)
    sync()
    fwd_matmul_qk = time.time() - t0

    t0 = time.time()
    attn_probs = F.softmax(attn_scores, dim=-1)
    sync()
    fwd_softmax = time.time() - t0

    t0 = time.time()
    out = torch.einsum("bhqk,bhkd->bhqd", attn_probs, V)
    sync()
    fwd_matmul_av = time.time() - t0

    del out, attn_probs, attn_scores
    return {"fwd_matmul_qk": fwd_matmul_qk, "fwd_softmax": fwd_softmax, "fwd_matmul_av": fwd_matmul_av}


def profile_full_backward(Q, K, V):
    """Profile full attention backward per-stage using autograd hooks.
    Q, K, V must have requires_grad=True."""
    # Forward (untimed) — builds autograd graph
    attn_scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(d_K)
    attn_probs = F.softmax(attn_scores, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", attn_probs, V)

    # Register hooks that fire in reverse order during backward:
    #   out hook → (matmul AV bwd) → attn_probs hook → (softmax bwd) → attn_scores hook → (matmul QK bwd) → end
    hook_times = []

    def record_time(grad):
        sync()
        hook_times.append(time.time())

    out.register_hook(record_time)
    attn_probs.register_hook(record_time)
    attn_scores.register_hook(record_time)

    grad_output = torch.ones_like(out)
    sync()
    out.backward(grad_output)
    sync()
    t_bwd_end = time.time()

    bwd_matmul_av = hook_times[1] - hook_times[0]
    bwd_softmax = hook_times[2] - hook_times[1]
    bwd_matmul_qk = t_bwd_end - hook_times[2]

    return {"bwd_matmul_av": bwd_matmul_av, "bwd_softmax": bwd_softmax, "bwd_matmul_qk": bwd_matmul_qk}


# ============================================================
# k-MIP Attention: forward (per-stage) + backward (single block)
# ============================================================

def profile_kmip_forward(Q, K, V):
    """Profile k-MIP forward pass per-stage. Returns dict of stage times."""
    sync()

    # Stage 1: top-k search
    t0 = time.time()
    nearest_key_indices = symbolic_sparse_nearest_k_keys(Q, K, k).to(torch.int64)
    sync()
    fwd_topk = time.time() - t0

    # Stage 2: gather keys & values
    t0 = time.time()
    nearest_keys = torch.gather(
        input=K.unsqueeze(-2).expand(*K.shape[:-1], k, d_K),
        dim=-3,
        index=nearest_key_indices.unsqueeze(-1).expand(*nearest_key_indices.shape, d_K),
    )
    nearest_values = torch.gather(
        input=V.unsqueeze(-2).expand(*K.shape[:-1], k, val_dim),
        dim=-3,
        index=nearest_key_indices.unsqueeze(-1).expand(*nearest_key_indices.shape, val_dim),
    )
    sync()
    fwd_gather = time.time() - t0

    # Stage 3: softmax (QK scores on gathered keys + softmax)
    t0 = time.time()
    queries_extended = Q.unsqueeze(-2).expand(*Q.shape[:-1], k, d_K)
    attn_weights = (queries_extended * nearest_keys).sum(-1) / math.sqrt(d_K)
    attn_weights = F.softmax(attn_weights, dim=-1)
    sync()
    fwd_softmax = time.time() - t0

    # Stage 4: matmul (weighted sum of gathered values)
    t0 = time.time()
    out = (attn_weights.unsqueeze(-1) * nearest_values).sum(dim=-2)
    sync()
    fwd_matmul = time.time() - t0

    del out, attn_weights, queries_extended, nearest_keys, nearest_values, nearest_key_indices

    return {
        "fwd_topk": fwd_topk,
        "fwd_gather": fwd_gather,
        "fwd_softmax": fwd_softmax,
        "fwd_matmul": fwd_matmul,
    }


def profile_kmip_backward(Q, K, V):
    """Run full k-MIP forward (untimed), then time backward.
    Backward only flows through gather+softmax+matmul (top-k is @torch.no_grad)."""
    # Full forward (untimed)
    nearest_key_indices = symbolic_sparse_nearest_k_keys(Q, K, k).to(torch.int64)

    nearest_keys = torch.gather(
        input=K.unsqueeze(-2).expand(*K.shape[:-1], k, d_K),
        dim=-3,
        index=nearest_key_indices.unsqueeze(-1).expand(*nearest_key_indices.shape, d_K),
    )
    nearest_values = torch.gather(
        input=V.unsqueeze(-2).expand(*K.shape[:-1], k, val_dim),
        dim=-3,
        index=nearest_key_indices.unsqueeze(-1).expand(*nearest_key_indices.shape, val_dim),
    )
    queries_extended = Q.unsqueeze(-2).expand(*Q.shape[:-1], k, d_K)
    attn_weights = (queries_extended * nearest_keys).sum(-1) / math.sqrt(d_K)
    attn_weights = F.softmax(attn_weights, dim=-1)
    out = (attn_weights.unsqueeze(-1) * nearest_values).sum(dim=-2)

    grad_output = torch.ones_like(out)
    sync()

    # Time backward
    t0 = time.time()
    out.backward(grad_output)
    sync()
    bwd_time = time.time() - t0

    del out, attn_weights, queries_extended, nearest_keys, nearest_values, nearest_key_indices, grad_output

    return bwd_time


# ============================================================
# Main profiling loop
# ============================================================

def make_tensors(requires_grad):
    Q = torch.randn(B, H, N, d_K, device=DEVICE, dtype=DTYPE, requires_grad=requires_grad)
    K = torch.randn(B, H, N, d_K, device=DEVICE, dtype=DTYPE, requires_grad=requires_grad)
    V = torch.randn(B, H, N, val_dim, device=DEVICE, dtype=DTYPE, requires_grad=requires_grad)
    return Q, K, V


print(f"Per-stage profiling: N={N}, k={k}, d_K={d_K}, val_dim={val_dim}")
print(f"Warmup runs: {NUM_WARMUP}, measurement runs: {NUM_RUNS}")
print()

# --- Profile Full Attention ---
print("=== Full Attention ===")
full_runs = []
for run in range(NUM_WARMUP + NUM_RUNS):
    # Forward (per-stage timing, requires_grad=True to match profiling-experiment.py)
    Q, K_t, V = make_tensors(requires_grad=True)
    fwd_result = profile_full_forward(Q, K_t, V)
    del Q, K_t, V

    # Backward (per-stage via hooks, separate forward builds the graph)
    Q, K_t, V = make_tensors(requires_grad=True)
    bwd_result = profile_full_backward(Q, K_t, V)
    del Q, K_t, V

    result = {**fwd_result, **bwd_result}
    if run >= NUM_WARMUP:
        full_runs.append(result)
        print(f"  Run {run - NUM_WARMUP}: { {k: f'{v*1000:.2f}ms' for k, v in result.items()} }")

# --- Profile k-MIP Attention ---
print("\n=== k-MIP Attention ===")
kmip_fwd_runs = []
kmip_bwd_runs = []

for run in range(NUM_WARMUP + NUM_RUNS):
    # Forward (per-stage)
    Q, K_t, V = make_tensors(requires_grad=True)
    fwd_result = profile_kmip_forward(Q, K_t, V)
    del Q, K_t, V

    # Backward (single block)
    Q, K_t, V = make_tensors(requires_grad=True)
    bwd_time = profile_kmip_backward(Q, K_t, V)
    del Q, K_t, V

    if run >= NUM_WARMUP:
        kmip_fwd_runs.append(fwd_result)
        kmip_bwd_runs.append(bwd_time)
        print(f"  Run {run - NUM_WARMUP}: fwd={ {k: f'{v*1000:.2f}ms' for k, v in fwd_result.items()} }, bwd={bwd_time*1000:.2f}ms")


# ============================================================
# Aggregate results (mean over runs, convert to ms)
# ============================================================

def mean_of_key(runs, key):
    return np.mean([r[key] for r in runs]) * 1000  # ms

# Full attention stages (bottom-to-top order in the stacked bar)
full_fwd_stages = [
    ("Attention matrix",  mean_of_key(full_runs, "fwd_matmul_qk")),
    ("Softmax",           mean_of_key(full_runs, "fwd_softmax")),
    ("Matmul",            mean_of_key(full_runs, "fwd_matmul_av")),
]
full_bwd_stages = [
    ("Matmul\n(backward)",            mean_of_key(full_runs, "bwd_matmul_av")),
    ("Softmax\n(backward)",           mean_of_key(full_runs, "bwd_softmax")),
    ("Attention matrix\n(backward)",  mean_of_key(full_runs, "bwd_matmul_qk")),
]

# k-MIP stages
kmip_fwd_stages = [
    ("Top-k search",   mean_of_key(kmip_fwd_runs, "fwd_topk")),
    ("Gather values",  mean_of_key(kmip_fwd_runs, "fwd_gather")),
    ("Softmax",        mean_of_key(kmip_fwd_runs, "fwd_softmax")),
    ("Matmul",         mean_of_key(kmip_fwd_runs, "fwd_matmul")),
]
kmip_bwd_time = np.mean(kmip_bwd_runs) * 1000  # ms
kmip_bwd_stages = [
    ("Backward", kmip_bwd_time),
]

print("\n=== Summary (ms) ===")
print("Full fwd:", full_fwd_stages)
print("Full bwd:", full_bwd_stages)
print("Full total:", sum(t for _, t in full_fwd_stages) + sum(t for _, t in full_bwd_stages))
print("k-MIP fwd:", kmip_fwd_stages)
print("k-MIP bwd:", kmip_bwd_stages)
print("k-MIP total:", sum(t for _, t in kmip_fwd_stages) + sum(t for _, t in kmip_bwd_stages))


# ============================================================
# Plot
# ============================================================

GOLD = "#FFD700"
LIGHT_BLUE = "#87CEEB"
EDGE_COLOR = "black"
TEXT_COLOR = "black"
from matplotlib.patches import Patch, FancyBboxPatch, ConnectionPatch

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.05

full_total = sum(t for _, t in full_fwd_stages) + sum(t for _, t in full_bwd_stages)
kmip_total = sum(t for _, t in kmip_fwd_stages) + sum(t for _, t in kmip_bwd_stages)

x_positions = [-0.10, 0.03]  # Full, k-MIP


def draw_stacked_bar(target_ax, x, fwd_stages, bwd_stages, bw,
                     label_side=None, fontsize=8):
    """Draw one stacked bar on target_ax. Returns bar top."""
    bottom = 0.0
    segments = []

    for name, val in fwd_stages:
        target_ax.bar(x, val, bw, bottom=bottom,
                      color=GOLD, edgecolor=EDGE_COLOR, linewidth=1.2)
        segments.append((bottom, bottom + val, name, val))
        bottom += val

    for name, val in bwd_stages:
        target_ax.bar(x, val, bw, bottom=bottom,
                      color=LIGHT_BLUE, edgecolor=EDGE_COLOR, linewidth=1.2)
        segments.append((bottom, bottom + val, name, val))
        bottom += val

    if label_side is None:
        return bottom

    # All labels placed outside the bar with connecting lines
    min_gap = bottom * 0.03
    positions = [(b + t) / 2 for b, t, _, _ in segments]
    # Spread positions to avoid overlap (bottom-to-top)
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] < min_gap:
            positions[i] = positions[i - 1] + min_gap

    x_offset = 0.036
    for (b, t, name, val), text_y in zip(segments, positions):
        cy = (b + t) / 2
        if label_side == "right":
            target_ax.annotate(
                name, xy=(x + bw / 2, cy),
                xytext=(x + bw / 2 + x_offset, text_y),
                fontsize=fontsize, fontweight="bold", color=TEXT_COLOR,
                va="center", ha="left", multialignment="center",
                arrowprops=dict(arrowstyle="-", color=TEXT_COLOR, lw=0.8),
            )
        else:
            target_ax.annotate(
                name, xy=(x - bw / 2, cy),
                xytext=(x - bw / 2 - x_offset, text_y),
                fontsize=fontsize, fontweight="bold", color=TEXT_COLOR,
                va="center", ha="right", multialignment="center",
                arrowprops=dict(arrowstyle="-", color=TEXT_COLOR, lw=0.8),
            )

    return bottom


# --- Main chart ---
draw_stacked_bar(ax, x_positions[0], full_fwd_stages, full_bwd_stages, bar_width,
                 label_side="left")
# Aggregate k-MIP into single forward/backward segments for the main chart
# (individual stages are too thin and their edges make the bar look dark)
kmip_fwd_total = [("Forward", sum(t for _, t in kmip_fwd_stages))]
kmip_bwd_total = [("Backward", sum(t for _, t in kmip_bwd_stages))]
draw_stacked_bar(ax, x_positions[1], kmip_fwd_total, kmip_bwd_total, bar_width)

ax.set_xticks(x_positions)
ax.set_xticklabels(["Full", "k-MIP"], fontsize=12, fontweight="bold")
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_xlim(-0.35, 0.40)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_elements = [
    Patch(facecolor=GOLD, edgecolor=EDGE_COLOR, label="Forward"),
    Patch(facecolor=LIGHT_BLUE, edgecolor=EDGE_COLOR, label="Backward"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
ax.set_title(r"Per-stage profiling: Full vs k-MIP Attention ($N=10^{4.5}$)", fontsize=13)

# --- Inset zoom panel for k-MIP ---
# Position inset so its left edge is offset right of the k-MIP bar's right edge
fig.canvas.draw()  # needed to get accurate coordinate transforms
kmip_right_fig = ax.transData.transform((x_positions[1] + bar_width / 2, 0))[0]
kmip_right_fig /= fig.get_size_inches()[0] * fig.dpi  # convert to figure fraction
inset_gap = 0.04  # gap between k-MIP bar and inset in figure fraction
inset_left = kmip_right_fig + inset_gap
inset_width = 0.95 - inset_left  # extend to near right edge of figure
inset_height = 0.45
main_pos = ax.get_position()
inset_bottom = main_pos.y0  # align with main chart bottom

inset_ax = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])

inset_bw = bar_width  # same width as main bars
# Set inset xlim so bar looks same visual width as in main chart
main_xlim_range = 0.40 - (-0.35)  # ax xlim range
main_axes_width = main_pos.width
inset_visual_ratio = inset_width / main_axes_width
inset_xlim_range = main_xlim_range * inset_visual_ratio
inset_ax.set_xlim(-inset_xlim_range * 0.30, inset_xlim_range * 0.70)

# Draw bars without auto-labels, then place labels manually to avoid overlap
draw_stacked_bar(inset_ax, 0, kmip_fwd_stages, kmip_bwd_stages, inset_bw)

# Compute segment boundaries for manual annotation
kmip_segments = []  # (bottom, top, name)
_bot = 0.0
for name, val in kmip_fwd_stages + kmip_bwd_stages:
    kmip_segments.append((_bot, _bot + val, name))
    _bot += val

# Manual label positions: Top-k gets auto, then Gather/Softmax/Matmul/Backward
# are spaced downward from the Matmul segment center
_x_off = 0.036
_fs = 8
# Top-k search: beside its segment (large, no overlap issue)
_b, _t, _n = kmip_segments[0]
inset_ax.annotate(_n, xy=(inset_bw / 2, (_b + _t) / 2),
    xytext=(inset_bw / 2 + _x_off, (_b + _t) / 2),
    fontsize=_fs, fontweight="bold", color=TEXT_COLOR,
    va="center", ha="left",
    arrowprops=dict(arrowstyle="-", color=TEXT_COLOR, lw=0.8))

# Matmul (idx 3): beside its segment
_b, _t, _n = kmip_segments[3]
_matmul_y = (_b + _t) / 2
inset_ax.annotate(_n, xy=(inset_bw / 2, _matmul_y),
    xytext=(inset_bw / 2 + _x_off, _matmul_y),
    fontsize=_fs, fontweight="bold", color=TEXT_COLOR,
    va="center", ha="left",
    arrowprops=dict(arrowstyle="-", color=TEXT_COLOR, lw=0.8))

# Softmax (idx 2): slightly below Matmul
_b, _t, _n = kmip_segments[2]
_softmax_y = _matmul_y - kmip_total * 0.06
inset_ax.annotate(_n, xy=(inset_bw / 2, (_b + _t) / 2),
    xytext=(inset_bw / 2 + _x_off, _softmax_y),
    fontsize=_fs, fontweight="bold", color=TEXT_COLOR,
    va="center", ha="left",
    arrowprops=dict(arrowstyle="-", color=TEXT_COLOR, lw=0.8,
                    relpos=(0, 0.5)))

# Gather values (idx 1): slightly below Softmax
_b, _t, _n = kmip_segments[1]
_gather_y = _softmax_y - kmip_total * 0.06
inset_ax.annotate(_n, xy=(inset_bw / 2, (_b + _t) / 2),
    xytext=(inset_bw / 2 + _x_off, _gather_y),
    fontsize=_fs, fontweight="bold", color=TEXT_COLOR,
    va="center", ha="left",
    arrowprops=dict(arrowstyle="-", color=TEXT_COLOR, lw=0.8,
                    relpos=(0, 0.5)))

# Backward (idx 4): beside its segment
_b, _t, _n = kmip_segments[4]
inset_ax.annotate(_n, xy=(inset_bw / 2, (_b + _t) / 2),
    xytext=(inset_bw / 2 + _x_off, (_b + _t) / 2),
    fontsize=_fs, fontweight="bold", color=TEXT_COLOR,
    va="center", ha="left",
    arrowprops=dict(arrowstyle="-", color=TEXT_COLOR, lw=0.8))

inset_ax.set_ylim(0, kmip_total * 1.15)
inset_ax.set_xticks([0])
inset_ax.set_xticklabels(["k-MIP"], fontsize=12, fontweight="bold")
inset_ax.set_ylabel("Time (ms)", fontsize=9)
inset_ax.set_title("k-MIP (zoomed)", fontsize=10, fontweight="bold", color=TEXT_COLOR)
inset_ax.spines["top"].set_visible(False)
inset_ax.spines["right"].set_visible(False)
inset_ax.patch.set_facecolor("#f7f7f7")
inset_ax.patch.set_alpha(0.8)
# Remove the 0 tick from the y-axis
yticks = [t for t in inset_ax.get_yticks() if t > 0]
inset_ax.set_yticks(yticks)

# --- Connecting dashed lines: k-MIP bar (main) → k-MIP bar (inset) ---
# Bottom: right-bottom of main bar → left-bottom of inset bar
con_bottom = ConnectionPatch(
    xyA=(x_positions[1] + bar_width / 2, 0),
    xyB=(0 - inset_bw / 2, 0),
    coordsA="data", coordsB="data",
    axesA=ax, axesB=inset_ax,
    color=TEXT_COLOR, linestyle="--", linewidth=1.0, alpha=0.5,
)
fig.add_artist(con_bottom)

# Top: right-top of main bar → left-top of inset bar
con_top = ConnectionPatch(
    xyA=(x_positions[1] + bar_width / 2, kmip_total),
    xyB=(0 - inset_bw / 2, kmip_total),
    coordsA="data", coordsB="data",
    axesA=ax, axesB=inset_ax,
    color=TEXT_COLOR, linestyle="--", linewidth=1.0, alpha=0.5,
)
fig.add_artist(con_top)

plt.savefig("per-stage-profiling.png", dpi=150, bbox_inches="tight")
plt.savefig("per-stage-profiling.pdf", bbox_inches="tight")
print("\nSaved per-stage-profiling.png and per-stage-profiling.pdf")
