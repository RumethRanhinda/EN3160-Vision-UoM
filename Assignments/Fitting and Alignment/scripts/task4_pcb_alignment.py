"""
Task 4: Feature-Based PCB Alignment and Defect/Difference Inspection
Course: EN3160 Image Processing and Machine Vision
Student: Samarasinghe S.M.R.R. (230566U)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_pcb(template_bgr, target_bgr, ratio_thresh=0.75, ransac_thresh=5.0):
    """
    Align a target PCB image to the template using SIFT, BFMatcher, and RANSAC homography.
    """
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    h_t, w_t = template_gray.shape

    # 1. Detect SIFT features
    sift = cv2.SIFT_create()
    kp_tgt, des_tgt = sift.detectAndCompute(target_gray, None)
    kp_tpl, des_tpl = sift.detectAndCompute(template_gray, None)

    # 2. Match features with BFMatcher and Lowe's ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_tgt, des_tpl, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good_matches) < 4:
        raise ValueError(f"Insufficient good matches found: {len(good_matches)}")

    src_pts = np.float32([kp_tgt[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_tpl[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 3. Find Homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    inliers = [good_matches[i] for i, val in enumerate(mask.ravel()) if val == 1]

    # 4. Warp target to template coordinate frame
    warped_bgr = cv2.warpPerspective(target_bgr, H, (w_t, h_t))
    
    # 5. Difference image & metrics on overlapping board region
    diff_bgr = cv2.absdiff(template_bgr, warped_bgr)
    diff_gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)
    
    mask_overlap = cv2.warpPerspective(np.ones_like(target_gray) * 255, H, (w_t, h_t)) > 200
    mean_diff = float(np.mean(diff_gray[mask_overlap]))

    return {
        'H': H,
        'kp_target': kp_tgt,
        'kp_template': kp_tpl,
        'good_matches': good_matches,
        'inliers': inliers,
        'warped_bgr': warped_bgr,
        'diff_gray': diff_gray,
        'mask_overlap': mask_overlap,
        'mean_diff': mean_diff
    }

def run_task4():
    pcb_dir = 'src/pcbs'
    template_path = os.path.join(pcb_dir, 'AECC1.jpg')
    template_bgr = cv2.imread(template_path)
    
    target_names = ['AAAA1.jpg', 'ADAD1.jpg', 'AECC2.jpg', 'AECC3.jpg', 'AECC4.jpg']
    results = {}

    print(f"{'Image':<12}{'Keypoints':<12}{'Good Matches':<14}{'RANSAC Inliers':<16}{'Inlier %':<12}{'Mean Diff':<10}")
    print("-" * 76)

    for name in target_names:
        target_bgr = cv2.imread(os.path.join(pcb_dir, name))
        res = align_pcb(template_bgr, target_bgr)
        results[name] = res
        inlier_pct = (len(res['inliers']) / len(res['good_matches'])) * 100
        print(f"{name:<12}{len(res['kp_target']):<12}{len(res['good_matches']):<14}{len(res['inliers']):<16}{inlier_pct:<12.1f}{res['mean_diff']:<10.2f}")

    # Plot figure: 5 rows x 2 cols
    fig, axes = plt.subplots(len(target_names), 2, figsize=(11, 14.5))

    for i, name in enumerate(target_names):
        res = results[name]
        target_bgr = cv2.imread(os.path.join(pcb_dir, name))
        
        # Draw top 35 inliers for clean match lines
        match_vis = cv2.drawMatches(target_bgr, res['kp_target'], template_bgr, res['kp_template'], 
                                    res['inliers'][:35], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        axes[i, 0].imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"{name} Matches with AECC1 ({len(res['inliers'])} inliers / {len(res['good_matches'])} good)", fontsize=9.5)
        axes[i, 0].axis('off')

        # Clean difference image
        disp_diff = res['diff_gray'].copy()
        disp_diff[~res['mask_overlap']] = 0
        im = axes[i, 1].imshow(disp_diff, cmap='inferno', vmin=0, vmax=120)
        axes[i, 1].set_title(f"{name} Difference Image (Mean: {res['mean_diff']:.1f})", fontsize=9.5)
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_task4()
