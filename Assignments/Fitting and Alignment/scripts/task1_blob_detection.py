import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


def detect_sunflower_blobs(
    image_path="src/the_berry_farms_sunflower_field.jpeg",
    sigma_min=1.5,
    sigma_max=8.0,
    num_scales=20,
    threshold=0.10,
):
    """
    Blob detection using Laplacian of Gaussians and scale-space extrema detection.
    Image dimensions: 360 x 360 (due to cv.IMREAD_REDUCED_COLOR_4).
    """
    # Read image
    im = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_4)
    if im is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    h, w = gray.shape

    # Build geometrically spaced scale space
    sigmas = np.geomspace(sigma_min, sigma_max, num_scales)
    scale_space = np.zeros((num_scales, h, w), dtype=np.float64)

    for i, s in enumerate(sigmas):
        lap = ndi.gaussian_laplace(gray, sigma=s)
        scale_space[i] = (s ** 2) * lap

    # 3D Non-Maximum Suppression across (scale, y, x)
    footprint = np.ones((3, 3, 3), dtype=bool)
    max_vol = ndi.maximum_filter(scale_space, footprint=footprint, mode="constant", cval=0.0)

    # Local maxima above threshold (excluding boundary scales)
    peaks = (scale_space == max_vol) & (scale_space > threshold)
    peaks[0, :, :] = False
    peaks[-1, :, :] = False

    peak_coords = np.argwhere(peaks)

    # Extract circle parameters (center (x, y) and radius r = sqrt(2) * sigma)
    blobs = []
    for s_idx, y, x in peak_coords:
        if 0 < x < (w - 1) and 0 < y < (h - 1):
            s = sigmas[s_idx]
            r = np.sqrt(2) * s
            blobs.append({
                "x": float(x),
                "y": float(y),
                "r": float(r),
                "sigma": float(s),
                "score": float(scale_space[s_idx, y, x]),
            })

    return im_rgb, sigmas, blobs


if __name__ == "__main__":
    im_rgb, sigmas, blobs = detect_sunflower_blobs(
        image_path="src/the_berry_farms_sunflower_field.jpeg",
        sigma_min=1.5,
        sigma_max=8.0,
        num_scales=20,
        threshold=0.10,
    )

    print(f"Total detected blobs: {len(blobs)}")
    print(f"Range of sigma values used: [{sigmas[0]:.2f}, {sigmas[-1]:.2f}] across {len(sigmas)} levels")
    print(f"Corresponding radius range: [{np.sqrt(2)*sigmas[0]:.2f} px, {np.sqrt(2)*sigmas[-1]:.2f} px]")

    # Sort circles by radius to find the largest circles
    largest_circles = sorted(blobs, key=lambda b: b["r"], reverse=True)

    print("\nParameters of the 5 Largest Circles:")
    for i, b in enumerate(largest_circles[:5]):
        print(f"  {i+1}: Center=({b['x']:.1f}, {b['y']:.1f}), Radius={b['r']:.2f} px, Sigma={b['sigma']:.2f}, LoG={b['score']:.4f}")

    # Plot detected circles on the sunflower field image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(im_rgb)

    # Draw all detected circles
    for b in blobs:
        c = plt.Circle((b["x"], b["y"]), b["r"], color="yellow", fill=False, linewidth=1.0, alpha=0.8)
        ax.add_patch(c)

    # Highlight top 5 largest circles in red
    for i, b in enumerate(largest_circles[:5]):
        c = plt.Circle((b["x"], b["y"]), b["r"], color="red", fill=False, linewidth=2.0)
        ax.add_patch(c)
        ax.text(
            b["x"], b["y"], f"#{i+1}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="red", alpha=0.9, edgecolor="none")
        )

    ax.set_title("Detected Sunflowers using Scale-Space LoG (Yellow=All, Red=Largest)", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
