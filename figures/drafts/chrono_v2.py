"""Chronophotograph of an EvoGym robot. Gait = simple sine controller found by random search (not PPO)."""
import numpy as np, gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import evogym.envs
from evogym import get_full_connectivity

BODY = np.array([[1,1,1,1,1],[3,3,3,3,3],[3,3,0,3,3],[4,4,0,4,4],[4,4,0,4,4]])
STEPS = 400
COLORS = {1: "#ECE8E1", 2: "#8A8F98", 3: "#FF8A3D", 4: "#2EC4D6"}  # rigid, soft, horiz. muscle, vert. muscle


def make():
    return gym.make("Walker-v0", body=BODY, connections=get_full_connectivity(BODY))


def gait(params, t, n):
    freq, amp = params[0], params[1]
    return np.clip(1.1 + amp * np.sin(freq * t + params[2:2 + n]), 0.6, 1.6)


def rollout(params, record_every=None):
    env = make(); env.reset(seed=0); u = env.unwrapped
    n = env.action_space.shape[0]
    pos = [u.object_pos_at_time(u.get_time(), "robot").copy()]
    for t in range(STEPS):
        env.step(gait(params, t, n))
        if record_every and (t + 1) % record_every == 0:
            pos.append(u.object_pos_at_time(u.get_time(), "robot").copy())
    dx = u.object_pos_at_time(u.get_time(), "robot")[0].mean() - pos[0][0].mean()
    env.close()
    return dx, pos


rng = np.random.default_rng(0)
n_act = make().action_space.shape[0]
best, best_dx = None, -np.inf
for _ in range(40):
    params = np.concatenate([[rng.uniform(0.1, 0.6), rng.uniform(0.2, 0.5)], rng.uniform(0, 2 * np.pi, n_act)])
    dx, _ = rollout(params)
    if dx > best_dx:
        best, best_dx = params, dx
print(f"best sine gait: dx = {best_dx:.2f} (lattice units of 0.1 -> {best_dx/0.1:.0f} voxels)")

_, all_frames = rollout(best, record_every=5)
# keep snapshots spaced by distance travelled (clearer than fixed time steps)
frames, last_x = [all_frames[0]], all_frames[0][0].mean()
for P in all_frames[1:]:
    if P[0].mean() - last_x >= 0.32:
        frames.append(P); last_x = P[0].mean()

# Map each point mass to its lattice index from the rest pose, then voxel corners.
p0 = frames[0]
ix = np.round(p0[0] / 0.1).astype(int); iy = np.round(p0[1] / 0.1).astype(int)
lookup = {(x, y): k for k, (x, y) in enumerate(zip(ix, iy))}
H, W = BODY.shape
x0, y0 = ix.min(), iy.min()
voxels = []
for r in range(H):
    for c in range(W):
        if BODY[r, c]:
            gx, gy = x0 + c, y0 + (H - 1 - r)
            corners = [lookup[(gx, gy)], lookup[(gx + 1, gy)], lookup[(gx + 1, gy + 1)], lookup[(gx, gy + 1)]]
            voxels.append((BODY[r, c], corners))

fig, ax = plt.subplots(figsize=(16, 4.2), dpi=150)
bg = "#0E1116"
fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
n_f = len(frames)
for i, P in enumerate(frames):
    alpha = 0.12 + 0.88 * (i / (n_f - 1)) ** 1.6
    for vtype, cs in voxels:
        ax.add_patch(Polygon(P[:, cs].T, closed=True, facecolor=COLORS[vtype], edgecolor=bg,
                             linewidth=0.6, alpha=alpha, joinstyle="round"))
com = np.array([P.mean(1) for P in frames])
ax.plot(com[:, 0], com[:, 1], color="#FFFFFF", alpha=0.25, lw=1, ls=(0, (2, 3)))
xmin, xmax = min(P[0].min() for P in frames) - 0.3, max(P[0].max() for P in frames) + 0.3
ax.axhline(0.1, color="#3A3F48", lw=2)
ax.fill_between([xmin, xmax], 0, 0.1, color="#161A21")
ax.set_xlim(xmin, xmax); ax.set_ylim(0, 1.0); ax.set_aspect("equal"); ax.axis("off")
ax.text(xmin + 0.05, 0.92, "Walker-v0 · hand-designed body · sine gait (not trained yet)",
        color="#9AA0A8", fontsize=10, family="monospace", va="top")
plt.savefig("chronophoto_walker_v2.png", facecolor=bg, bbox_inches="tight", pad_inches=0.1)
print("saved", len(frames), "snapshots")
