import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


def detect_sunflower_blobs(
    image_path="src/the_berry_farms_sunflower_field.jpeg",
    sigma_min=1.2,
    sigma_max=8.0,
    num_scales=24,
    threshold=0.05,
):
    """
    Blob detection using Laplacian of Gaussians and scale-space extrema detection.
    Image dimensions: 360 x 360 (due to cv.IMREAD_REDUCED_COLOR_4).
    """
    im = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_4)
    if im is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, w = gray.shape

    # Sunflower yellow petals mask (high warm saturation and brightness)
    yellow_petals = (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 40)

    # Build geometrically spaced scale-space LoG
    sigmas = np.geomspace(sigma_min, sigma_max, num_scales)
    scale_space = np.zeros((num_scales, h, w), dtype=np.float64)

    for i, s in enumerate(sigmas):
        lap = ndi.gaussian_laplace(gray, sigma=s)
        scale_space[i] = (s ** 2) * lap

    # 3D Non-Maximum Suppression across (scale, y, x)
    footprint = np.ones((3, 3, 3), dtype=bool)
    max_vol = ndi.maximum_filter(scale_space, footprint=footprint, mode="constant", cval=0.0)

    # Candidate peaks above threshold, excluding scale boundaries and non-field sky/trees (y < 175)
    peaks = (scale_space == max_vol) & (scale_space > threshold)
    peaks[0, :, :] = False
    peaks[-1, :, :] = False
    peaks[:, :175, :] = False

    peak_coords = np.argwhere(peaks)

    # Filter candidate blobs to confirm genuine sunflowers
    blobs = []
    for s_idx, y, x in peak_coords:
        if 0 < x < (w - 1) and 0 < y < (h - 1):
            s = sigmas[s_idx]
            blob_r = np.sqrt(2) * s
            score = scale_space[s_idx, y, x]

            # Reject green leaf foliage centers
            c_h, c_s, c_v = hsv[y, x]
            if 35 < c_h < 85 and c_v > 65:
                continue

            # Check surrounding annulus [0.7*r, 2.2*r] for yellow petals
            r_in = max(1.0, 0.7 * blob_r)
            r_out = min(25.0, 2.2 * blob_r)

            x_min = max(0, int(x - r_out))
            x_max = min(w, int(x + r_out + 1))
            y_min = max(0, int(y - r_out))
            y_max = min(h, int(y + r_out + 1))

            Y_sub, X_sub = np.ogrid[y_min:y_max, x_min:x_max]
            d2 = (X_sub - x) ** 2 + (Y_sub - y) ** 2
            annulus = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)

            petal_cnt = np.sum(yellow_petals[y_min:y_max, x_min:x_max] & annulus)
            ann_area = np.sum(annulus)
            petal_frac = petal_cnt / max(1, ann_area)

            # Small background blooms need minimal petal support; foreground blooms require robust petal rings
            if blob_r <= 3.5:
                if petal_cnt >= 1 or petal_frac >= 0.03:
                    blobs.append({"x": float(x), "y": float(y), "r": float(blob_r), "sigma": float(s), "score": float(score)})
            else:
                if petal_cnt >= 8 and petal_frac >= 0.20 and score >= 0.20:
                    blobs.append({"x": float(x), "y": float(y), "r": float(blob_r), "sigma": float(s), "score": float(score)})

    # Select top 5 largest sunflowers using spatial non-maximum suppression
    sorted_blobs = sorted(blobs, key=lambda b: (b["r"], b["score"]), reverse=True)
    largest_sunflowers = []
    for b in sorted_blobs:
        too_close = any(np.hypot(b["x"] - k["x"], b["y"] - k["y"]) < k["r"] * 0.7 for k in largest_sunflowers)
        if not too_close:
            largest_sunflowers.append(b)
        if len(largest_sunflowers) == 5:
            break

    return im_rgb, sigmas, blobs, largest_sunflowers


if __name__ == "__main__":
    im_rgb, sigmas, blobs, largest_circles = detect_sunflower_blobs(
        image_path="src/the_berry_farms_sunflower_field.jpeg",
        sigma_min=1.2,
        sigma_max=8.0,
        num_scales=24,
        threshold=0.05,
    )

    print(f"Total detected blobs: {len(blobs)}")
    print(f"Sigma range used: [{sigmas[0]:.2f}, {sigmas[-1]:.2f}] across {len(sigmas)} levels")
    print(f"Corresponding radius range: [{np.sqrt(2)*sigmas[0]:.2f} px, {np.sqrt(2)*sigmas[-1]:.2f} px]")

    print("\nParameters of the 5 Largest Circles:")
    print(f"{'Rank':<6} {'Center (x, y)':<18} {'Radius r (px)':<15} {'Scale sigma':<12} {'LoG Response':<12}")
    print("-" * 65)
    for i, b in enumerate(largest_circles):
        print(f"{i+1:<6} ({b['x']:.1f}, {b['y']:.1f}){'':<6} {b['r']:<15.2f} {b['sigma']:<12.2f} {b['score']:<12.4f}")

    # Plot detected circles on the sunflower field image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(im_rgb)

    # Draw all detected circles in yellow
    for b in blobs:
        c = plt.Circle((b["x"], b["y"]), b["r"], color="yellow", fill=False, linewidth=0.8, alpha=0.75)
        ax.add_patch(c)

    # Highlight top 5 largest circles in red
    for i, b in enumerate(largest_circles):
        c = plt.Circle((b["x"], b["y"]), b["r"], color="red", fill=False, linewidth=2.0)
        ax.add_patch(c)
        ax.text(
            b["x"], b["y"], f"#{i+1}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="red", alpha=0.9, edgecolor="none")
        )

    ax.set_title("Detected Sunflowers using Scale-Space LoG (Yellow=All, Red=5 Largest)", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
