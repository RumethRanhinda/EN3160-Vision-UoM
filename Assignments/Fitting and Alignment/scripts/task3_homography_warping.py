"""
Task 3: Planar Homography and Image Warping
Course: EN3160 Image Processing and Machine Vision
Student: Samarasinghe S.M.R.R. (230566U)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def superimpose_image(target_path, overlay_path, dst_pts, blend_mode='additive', alpha=0.5, blur=None):
    target = cv2.imread(target_path)
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if blur is not None:
        overlay = cv2.GaussianBlur(overlay, blur[0], blur[1])
    h_ov, w_ov = overlay.shape[:2]
    src_pts = np.float32([[0, 0], [w_ov - 1, 0], [w_ov - 1, h_ov - 1], [0, h_ov - 1]])
    
    H = cv2.getPerspectiveTransform(src_pts, dst_pts.astype(np.float32))
    h_t, w_t = target.shape[:2]
    warped = cv2.warpPerspective(overlay, H, (w_t, h_t))
    
    mask = np.zeros((h_t, w_t), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
    
    result = target.copy()
    if blend_mode == 'additive':
        result[mask > 0] = cv2.add(result[mask > 0], (warped[mask > 0, :3] * alpha).astype(np.uint8))
    elif blend_mode == 'replace':
        result[mask > 0] = warped[mask > 0, :3]
    return result, H

def run_task3():
    # Pair 1: Wadham College Left Facade + UK Flag
    pts_wadham = np.float32([[147, 208], [517, 292], [521, 521], [135, 519]])
    res_wadham, H1 = superimpose_image('src/wadham.jpg', 'src/flag.png', pts_wadham, blend_mode='additive', alpha=0.5)

    orig_wadham = cv2.imread('src/wadham.jpg')
    for i, pt in enumerate(pts_wadham):
        cv2.circle(orig_wadham, tuple(pt.astype(int)), 6, (0, 0, 255), -1)
        cv2.putText(orig_wadham, f"P{i+1}", (int(pt[0]) + 8, int(pt[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.polylines(orig_wadham, [pts_wadham.astype(np.int32)], True, (0, 255, 255), 2)

    # Pair 2: Living Room TV Screen + Blurred F1 Crash
    pts_tv = np.float32([[283, 19], [516, 19], [516, 178], [283, 176]])
    res_tv, H2 = superimpose_image('src/tv.jpg', 'src/crash.jpg', pts_tv, blend_mode='replace', blur=((5, 5), 1.5))

    orig_tv = cv2.imread('src/tv.jpg')
    for i, pt in enumerate(pts_tv):
        cv2.circle(orig_tv, tuple(pt.astype(int)), 4, (0, 0, 255), -1)
        cv2.putText(orig_tv, f"P{i+1}", (int(pt[0]) + 6, int(pt[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.polylines(orig_tv, [pts_tv.astype(np.int32)], True, (0, 255, 255), 2)

    print("Pair 1 Homography Matrix (UK Flag -> Wadham):")
    print(np.array2string(H1, precision=4, suppress_small=True))
    print("\nPair 2 Homography Matrix (F1 Crash -> TV Screen):")
    print(np.array2string(H2, precision=4, suppress_small=True))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes[0, 0].imshow(cv2.cvtColor(orig_wadham, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Pair 1: Wadham Left Facade (4 Corners)", fontsize=10); axes[0, 0].axis('off')
    axes[0, 1].imshow(cv2.cvtColor(res_wadham, cv2.COLOR_BGR2RGB)); axes[0, 1].set_title("Pair 1: Superimposed UK Flag (Fig. 2 Replication)", fontsize=10); axes[0, 1].axis('off')
    axes[1, 0].imshow(cv2.cvtColor(orig_tv, cv2.COLOR_BGR2RGB)); axes[1, 0].set_title("Pair 2: Living Room TV Screen (4 Corners)", fontsize=10); axes[1, 0].axis('off')
    axes[1, 1].imshow(cv2.cvtColor(res_tv, cv2.COLOR_BGR2RGB)); axes[1, 1].set_title("Pair 2: Superimposed F1 Crash (with Gaussian Blur)", fontsize=10); axes[1, 1].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_task3()
