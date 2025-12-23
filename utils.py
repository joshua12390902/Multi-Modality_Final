import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- 1. LPIPS 相關導入 (知覺品質指標) ---
import torch
import lpips 

LPIPS_MODEL = None
try:
    LPIPS_MODEL = lpips.LPIPS(net='alex').eval() 
except Exception:
    LPIPS_MODEL = None


# --- 影像對齊函式 ---
def align_images_orb_ransac(img_target_bgr, img_source_bgr):
    """ 使用 ORB 特徵點和 RANSAC 來自動對齊兩張圖片。 """
    img_target_gray = cv2.cvtColor(img_target_bgr, cv2.COLOR_BGR2GRAY)
    img_source_gray = cv2.cvtColor(img_source_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(img_source_gray, None)
    kp2, des2 = orb.detectAndCompute(img_target_gray, None)
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return img_source_bgr

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    
    if len(matches) < 4:
        return img_source_bgr
        
    pts_source = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_target = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(pts_source, pts_target, cv2.RANSAC, 5.0)
    
    height, width, _ = img_target_bgr.shape
    img_aligned_bgr = cv2.warpPerspective(img_source_bgr, H, (width, height))
    
    return img_aligned_bgr


# --- LPIPS 專用計算函式 ---
def calculate_lpips(image_ref, image_test):
    global LPIPS_MODEL
    
    if LPIPS_MODEL is None:
        return np.nan 

    ref = image_ref.astype(np.float32)
    test = image_test.astype(np.float32)
    
    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    test_rgb = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    
    ref_tensor = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0)
    test_tensor = torch.from_numpy(test_rgb).permute(2, 0, 1).unsqueeze(0)
    
    ref_tensor = ref_tensor / 127.5 - 1
    test_tensor = test_tensor / 127.5 - 1
    
    with torch.no_grad():
        lpips_value = LPIPS_MODEL(ref_tensor, test_tensor).item()
        
    return lpips_value


# --- 量化指標計算函式 (只計算 PSNR, SSIM, LPIPS) ---
def calculate_metrics(image_ref, image_test):
    """
    計算 PSNR, SSIM, 和 LPIPS 三個指標。
    """
    if image_ref.shape != image_test.shape:
        # 僅返回三個指標
        return 0.0, 0.0, np.nan 

    image_ref = image_ref.astype(np.uint8)
    image_test = image_test.astype(np.uint8)
    
    ref_gray = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

    psnr_value = psnr(ref_gray, test_gray, data_range=255)
    ssim_value = ssim(image_ref, image_test, data_range=255, channel_axis=2, multichannel=True)
    
    lpips_value = calculate_lpips(image_ref, image_test)
    
    # 僅返回三個指標
    return psnr_value, ssim_value, lpips_value


def run_metric_comparison(img_ref_bgr, img_test_bgr, label):
    """運行計算並列印結果"""
    
    # 僅接收三個回傳值
    psnr_val, ssim_val, lpips_val = calculate_metrics(img_ref_bgr, img_test_bgr)
    
    print(f"--- {label} ---")
    print(f"PSNR (dB): {psnr_val:.4f}")
    print(f"SSIM: {ssim_val:.4f}")
    
    lpips_str = f"{lpips_val:.4f}" if not np.isnan(lpips_val) else "N/A (請安裝 lpips)"

    print(f"LPIPS: {lpips_str}")
    print("-" * 20)
    
    # 僅回傳三個指標
    return psnr_val, ssim_val, lpips_val