import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def generate_noisy_data(seed=0):
    np.random.seed(seed)
    N = 100
    half_n = N // 2
    r = 10
    x0_gt, y0_gt = 2, 3  # Circle center
    s = r / 16  # std = 0.625
    t = np.random.uniform(0, 2 * np.pi, half_n)
    n = s * np.random.randn(half_n)
    x = x0_gt + (r + n) * np.cos(t)
    y = y0_gt + (r + n) * np.sin(t)
    X_circ = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

    s = 1.0  # Line noise std = 1.0
    m, b = -1, 2
    x = np.linspace(-12, 12, half_n)
    y = m * x + b + s * np.random.randn(half_n)
    X_line = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))
    X = np.vstack((X_circ, X_line))
    
    gt_circle = (x0_gt, y0_gt, r)
    gt_line = (m, b)
    return X, X_circ, X_line, gt_circle, gt_line


def fit_line_2pts(p1, p2):
    """Estimates line parameters [a, b, d] from 2 points with ||[a, b]|| = 1."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norm = np.hypot(dx, dy)
    if norm < 1e-8:
        return None
    a = -dy / norm
    b = dx / norm
    d = a * p1[0] + b * p1[1]
    return np.array([a, b, d])


def refine_line_tls(points):
    """Total Least Squares line fitting on inliers using SVD."""
    p_mean = np.mean(points, axis=0)
    centered = points - p_mean
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # Unit normal vector
    a, b = normal[0], normal[1]
    d = a * p_mean[0] + b * p_mean[1]
    return np.array([a, b, d])


def ransac_line(X, thresh=2.0, max_iters=1000, min_consensus=35):
    """Estimates line via RANSAC and refines using TLS."""
    best_inliers = []
    best_sample = None
    best_sample_model = None

    for _ in range(max_iters):
        idx = np.random.choice(len(X), 2, replace=False)
        model = fit_line_2pts(X[idx[0]], X[idx[1]])
        if model is None:
            continue
        dists = np.abs(model[0] * X[:, 0] + model[1] * X[:, 1] - model[2])
        inliers = np.where(dists < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_sample = idx
            best_sample_model = model

    tls_model = refine_line_tls(X[best_inliers])
    return tls_model, best_sample_model, best_sample, best_inliers


def fit_circle_3pts(p1, p2, p3):
    """Estimates circle center (xc, yc) and radius r from 3 non-collinear points."""
    A = np.array([
        [2 * (p2[0] - p1[0]), 2 * (p2[1] - p1[1])],
        [2 * (p3[0] - p2[0]), 2 * (p3[1] - p2[1])]
    ])
    B = np.array([
        p2[0]**2 + p2[1]**2 - (p1[0]**2 + p1[1]**2),
        p3[0]**2 + p3[1]**2 - (p2[0]**2 + p2[1]**2)
    ])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    xc, yc = np.linalg.solve(A, B)
    r = np.hypot(p1[0] - xc, p1[1] - yc)
    return np.array([xc, yc, r])


def refine_circle_opt(inliers, init_params):
    """Refines circle parameters [xc, yc, r] on inliers using scipy.optimize.minimize."""
    def loss(params):
        xc, yc, r = params
        return np.sum((np.hypot(inliers[:, 0] - xc, inliers[:, 1] - yc) - r)**2)

    res = minimize(loss, init_params, method="Nelder-Mead")
    return res.x


def ransac_circle(X_remnant, thresh=1.5, max_iters=2000, min_consensus=30):
    """Estimates circle on remnant points via RANSAC and refines using scipy.optimize.minimize."""
    best_inliers = []
    best_sample = None
    best_sample_model = None

    for _ in range(max_iters):
        idx = np.random.choice(len(X_remnant), 3, replace=False)
        model = fit_circle_3pts(X_remnant[idx[0]], X_remnant[idx[1]], X_remnant[idx[2]])
        if model is None or model[2] > 50:
            continue
        dists = np.abs(np.hypot(X_remnant[:, 0] - model[0], X_remnant[:, 1] - model[1]) - model[2])
        inliers = np.where(dists < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_sample = idx
            best_sample_model = model

    opt_model = refine_circle_opt(X_remnant[best_inliers], best_sample_model)
    return opt_model, best_sample_model, best_sample, best_inliers


def main():
    X, X_circ, X_line, gt_circ, gt_line = generate_noisy_data(seed=0)
    os.makedirs("doc/Images", exist_ok=True)

    # 1. Fit line with RANSAC
    thresh_line = 2.0  # 2 * sigma_line (sigma_line = 1.0)
    best_fit_line, sample_line_model, sample_line_idx, line_inliers = ransac_line(
        X, thresh=thresh_line, max_iters=1000, min_consensus=35
    )

    # 2. Subtract line consensus
    remnant_mask = np.ones(len(X), dtype=bool)
    remnant_mask[line_inliers] = False
    remnant_indices = np.where(remnant_mask)[0]
    X_remnant = X[remnant_indices]

    # 3. Fit circle on remnant points with RANSAC & scipy.optimize.minimize
    thresh_circ = 1.5  # 2.4 * sigma_circ (sigma_circ = 0.625)
    best_fit_circle, sample_circ_model, sample_circ_rel_idx, circ_inliers_rel = ransac_circle(
        X_remnant, thresh=thresh_circ, max_iters=2000, min_consensus=30
    )
    sample_circ_idx = remnant_indices[sample_circ_rel_idx]
    circ_inliers = remnant_indices[circ_inliers_rel]

    print("=== Task 2 Results ===")
    print(f"Ground Truth Line: m = {gt_line[0]}, b = {gt_line[1]}")
    print(f"RANSAC Sample Line: a={sample_line_model[0]:.4f}, b={sample_line_model[1]:.4f}, d={sample_line_model[2]:.4f}")
    print(f"Best-fit Line (TLS): a={best_fit_line[0]:.4f}, b={best_fit_line[1]:.4f}, d={best_fit_line[2]:.4f}")
    print(f"Line consensus inliers: {len(line_inliers)} / 50")

    print(f"\nGround Truth Circle: center = ({gt_circ[0]}, {gt_circ[1]}), radius = {gt_circ[2]}")
    print(f"RANSAC Sample Circle: center = ({sample_circ_model[0]:.4f}, {sample_circ_model[1]:.4f}), radius = {sample_circ_model[2]:.4f}")
    print(f"Best-fit Circle (Optimized): center = ({best_fit_circle[0]:.4f}, {best_fit_circle[1]:.4f}), radius = {best_fit_circle[2]:.4f}")
    print(f"Circle consensus inliers: {len(circ_inliers)} / 50")

    # 4. Plot matching Figure 1
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # All points
    ax.scatter(X[:, 0], X[:, 1], color="#1f77b4", s=25, label="All points")

    # Line inliers
    ax.scatter(X[line_inliers, 0], X[line_inliers, 1], color="#bcbd22", s=30, label="Line inliers")

    # Best sample for line (2 points)
    ax.scatter(X[sample_line_idx, 0], X[sample_line_idx, 1], color="#9467bd", marker="x", s=60, linewidths=2, label="Best sample for line")

    # Circle inliers
    ax.scatter(X[circ_inliers, 0], X[circ_inliers, 1], color="#17becf", s=30, label="Circle inliers")

    # Best sample for circle (3 points)
    ax.scatter(X[sample_circ_idx, 0], X[sample_circ_idx, 1], color="red", s=40, zorder=5, label="Best sample for circle")

    # Ground truth circle
    circle_gt_patch = plt.Circle((gt_circ[0], gt_circ[1]), gt_circ[2], color="#2ca02c", fill=False, linewidth=1.5, label="Ground truth circle")
    ax.add_patch(circle_gt_patch)
    ax.plot(gt_circ[0], gt_circ[1], "+", color="#2ca02c", markersize=7)

    # RANSAC circle (sample estimate dashed pink, refined solid magenta)
    circle_sample_patch = plt.Circle((sample_circ_model[0], sample_circ_model[1]), sample_circ_model[2], color="pink", linestyle="--", fill=False, linewidth=1.2, label="Sample circle")
    circle_fit_patch = plt.Circle((best_fit_circle[0], best_fit_circle[1]), best_fit_circle[2], color="#e377c2", fill=False, linewidth=1.8, label="RANSAC circle")
    ax.add_patch(circle_sample_patch)
    ax.add_patch(circle_fit_patch)

    # Ground truth line
    x_range = np.array([-14, 14])
    y_gt_line = gt_line[0] * x_range + gt_line[1]
    ax.plot(x_range, y_gt_line, color="#8c564b", linestyle="--", linewidth=1.5, label="Ground truth line")

    # RANSAC line (TLS refined)
    a, b, d = best_fit_line
    y_ransac_line = (d - a * x_range) / b
    ax.plot(x_range, y_ransac_line, color="#d62728", linewidth=1.8, label="RANSAC line")

    ax.set_xlim(-15, 15)
    ax.set_ylim(-13, 17)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()

    out_fig = "doc/Images/task2_ransac_fitting.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_fig}")


if __name__ == "__main__":
    main()
