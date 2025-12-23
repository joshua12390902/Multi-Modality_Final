import cv2
import bm3d
import numpy as np

from modalities import (
    estimate_noise_structure_mask,
    get_gradient_magnitude_sobel,
    get_gradient_mask_canny,
    estimate_global_noise_magnitude, 
)

def enhance_luminance_clahe(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    對 BGR 影像的 HSV 亮度通道 (V) 做 CLAHE，提升暗部亮度與局部對比。
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    V_enhanced = clahe.apply(V)

    img_enhanced_hsv = cv2.merge([H, S, V_enhanced])
    img_enhanced_bgr = cv2.cvtColor(img_enhanced_hsv, cv2.COLOR_HSV2BGR)
    return img_enhanced_bgr


def bm3d_denoising_colored(img_bgr, sigma_psd=25.0):
    """
    使用 BM3D 演算法對彩色圖像進行去噪。(保持不變)
    """
    img_float = img_bgr.astype(np.float32) / 255.0
    img_rgb_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
    sigma_float = sigma_psd / 255.0
    
    try:
        img_denoised_rgb_float = bm3d.bm3d(
            img_rgb_float, 
            sigma_psd=sigma_float, 
            stage_arg=bm3d.BM3DStages.ALL_STAGES 
        )
    except Exception:
        img_denoised_rgb_float = img_rgb_float
    
    img_denoised_rgb_float = np.clip(img_denoised_rgb_float, 0, 1.0).astype(np.float32)
    img_denoised_bgr_float = cv2.cvtColor(img_denoised_rgb_float, cv2.COLOR_RGB2BGR)
    img_denoised_bgr = (img_denoised_bgr_float * 255).astype(np.uint8)
    
    return img_denoised_bgr


# ------------------------------------------------------------------
# V1 NLM 融合函式 (可切換/可調整權重參數)
# ------------------------------------------------------------------

def fusion_algorithm_weighted_v1_nlm(
    img_bgr,
    percentile_floor=1.0,
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    nlm_h_override=None, # h 參數覆蓋 (用於消融實驗)
    alpha=0.7, # M3 結構權重
    beta=0.3,  # M2 梯度權重
    gamma_w=1.2, # Gamma 調整
):
    """
    V1 版本核心融合：
    使用 NLM (Non-Local Means) 作為降噪核心，h 參數可切換/自適應，權重可調整。
    """

    I_base = enhance_luminance_clahe(
        img_bgr,
        clip_limit=clahe_clip,
        tile_grid_size=clahe_tile,
    )

    M3_structure = estimate_noise_structure_mask(
        img_bgr, percentile_floor=percentile_floor
    )
    M2_gradient = get_gradient_mask_canny(
        img_bgr
    )

    # ----------------------------------------------------
    # ** NLM 參數 h 計算 **
    # ----------------------------------------------------
    if nlm_h_override is not None:
        # 1. 使用固定值 (對照組)
        nlm_h_final = float(nlm_h_override)
        print(f"NLM 固定 h 值設定為: {nlm_h_final:.2f}")
    else:
        # 2. 運行自適應邏輯 (最終組)
        try:
            noise_mag = estimate_global_noise_magnitude(img_bgr)
        except NameError:
            nlm_h_final = 15.0
        else:
            # 臨時調整縮放因子為 5.0，以確保 h 值與 15.0 有差異
            nlm_h_scaling_factor = 5.0 
            
            nlm_h_adaptive = nlm_h_scaling_factor * noise_mag 
            nlm_h_final = float(np.clip(nlm_h_adaptive, 10.0, 50.0)) 
        
        print(f"NLM 自適應 h 值設定為: {nlm_h_final:.2f}")

    
    # Smooth branch：使用 NLM
    I_smooth = cv2.fastNlMeansDenoisingColored(
        I_base, None, nlm_h_final, nlm_h_final, 7, 21
    )

    # ----------------------------------------------------
    # ** 權重計算 (使用傳入的 alpha/beta/gamma_w) **
    # ----------------------------------------------------
    S = M3_structure.astype(np.float32) / 255.0
    G = M2_gradient.astype(np.float32) / 255.0

    # 關鍵：使用 alpha/beta 進行線性組合
    W = alpha * S + beta * G 
    W = np.clip(W, 0.0, 1.0)
    
    # 關鍵：使用 gamma_w 進行非線性調整
    W = W ** gamma_w 

    W_3ch = np.dstack([W, W, W])

    # 4. 加權融合
    I_base_f = I_base.astype(np.float32)
    I_smooth_f = I_smooth.astype(np.float32)

    I_enhanced_f = W_3ch * I_base_f + (1.0 - W_3ch) * I_smooth_f
    I_enhanced = np.clip(I_enhanced_f, 0, 255).astype(np.uint8)

    return I_enhanced

# ------------------------------------------------------------------
# V2 BM3D 融合函式 (保留)
# ------------------------------------------------------------------

def fusion_algorithm_weighted_v2(
    img_bgr,
    percentile_floor=1.0,
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    bm3d_sigma=25.0, 
):
    """
    V2：多模態加權融合 + BM3D 降噪核心
    """
    I_base = enhance_luminance_clahe(
        img_bgr,
        clip_limit=clahe_clip,
        tile_grid_size=clahe_tile,
    )

    M3_structure = estimate_noise_structure_mask(
        img_bgr, percentile_floor=percentile_floor
    )
    M2_gradient = get_gradient_mask_canny(
        img_bgr
    )

    I_smooth = bm3d_denoising_colored(I_base, sigma_psd=bm3d_sigma)

    S = M3_structure.astype(np.float32) / 255.0
    G = M2_gradient.astype(np.float32) / 255.0

    W = 0.7 * S + 0.3 * G
    W = np.clip(W, 0.0, 1.0)
    gamma_w = 1.2 
    W = W ** gamma_w
    W_3ch = np.dstack([W, W, W])

    I_base_f = I_base.astype(np.float32)
    I_smooth_f = I_smooth.astype(np.float32)

    I_enhanced_f = W_3ch * I_base_f + (1.0 - W_3ch) * I_smooth_f
    I_enhanced = np.clip(I_enhanced_f, 0, 255).astype(np.uint8)

    return I_enhanced


# ------------------------------------------------------------------
# 輔助 Log 函式 
# ------------------------------------------------------------------
import sys
import datetime

def log_print(message):
    """
    輸出到控制台，並將內容寫入 'log.txt'。
    """
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    full_message = timestamp + message
    print(full_message)

    try:
        with open('log.txt', 'a', encoding='utf-8') as f:
            f.write(full_message + '\n')
    except Exception as e:
        # 如果寫入失敗，至少保證控制台能看到
        print(f"!!! Error writing to log.txt: {e}")