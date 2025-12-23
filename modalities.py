import cv2
import numpy as np


# -------------------------------------------------------------
# 模態生成函式 (Modality Generation)
# -------------------------------------------------------------

def get_gradient_mask_canny(img_bgr, low_threshold=50, high_threshold=150):
    """
    M2 替代方案：使用 Canny 檢測邊緣，並進行擴散作為權重。
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Canny 檢測邊緣 (自動包含高斯平滑)
    edges = cv2.Canny(img_gray, low_threshold, high_threshold)
    
    # 2. 擴散邊緣以創建權重遮罩（從二值圖轉為連續權重）
    # 這裡使用高斯模糊來擴散邊緣，使邊緣周圍的權重從 1 逐漸降到 0
    # 這樣可以形成一個連續的權重梯度，比 Sobel 更細膩。
    kernel_size = 5 # 可調參數
    mask_weight = cv2.GaussianBlur(edges, (kernel_size, kernel_size), 0)
    
    # 最終正規化到 0-255
    mask_normalized = np.clip(mask_weight, 0, 255).astype(np.uint8)
    return mask_normalized



def get_gradient_magnitude_sobel(img_bgr):
    """
    M2: 計算 Sobel 梯度強度圖。
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 計算 X 和 Y 方向梯度
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 計算梯度強度
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # 正規化到 0-255 (uint8)
    magnitude_normalized = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude_normalized

def estimate_noise_structure_mask(img_bgr, percentile_floor=1.0, kernel_size=(5, 5)):
    """
    M3: 估計結構遮罩 (局部標準差減去噪點基線)。
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_float = img_gray.astype(np.float32)

    # 1. 計算局部標準差 (Local Standard Deviation)
    # 使用高斯模糊的標準差作為局部變異性的近似
    mu = cv2.GaussianBlur(img_float, kernel_size, 0)
    mu_sq = mu * mu
    sigma_sq = cv2.GaussianBlur(img_float * img_float, kernel_size, 0) - mu_sq
    
    # 標準差 (Sigma)
    sigma = np.sqrt(np.maximum(0, sigma_sq))

    # 2. 估計噪點基線 (Noise Floor Estimation)
    # 假設變異性最低的區域是純噪點，取 1% percentile
    noise_floor = np.percentile(sigma, percentile_floor) 

    # 3. 結構遮罩 M3
    # M3 = max(0, sigma - noise_floor)
    structure_mask = np.maximum(0, sigma - noise_floor)
    
    # 正規化到 0-255 (uint8)
    structure_mask_normalized = cv2.normalize(structure_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return structure_mask_normalized

# -------------------------------------------------------------
# 參數自適應所需函式
# -------------------------------------------------------------

def estimate_global_noise_magnitude(img_bgr):
    """
    估計圖像的整體噪點強度（作為 NLM 參數 h 的代理）。
    使用局部標準差的 5% 百分位數作為噪點基線的近似。
    
    輸出: 噪點標準差的估計值 (float)
    """
    # 1. 轉換為灰階
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. 使用 7x7 窗口計算局部標準差的近似值
    # 這裡可以沿用 estimate_noise_structure_mask 中的 sigma 計算邏輯
    img_float = img_gray.astype(np.float32)
    mu = cv2.GaussianBlur(img_float, (7, 7), 0)
    mu_sq = mu * mu
    sigma_sq = cv2.GaussianBlur(img_float * img_float, (7, 7), 0) - mu_sq
    sigma = np.sqrt(np.maximum(0, sigma_sq))
    
    # 3. 估計噪點基線：取局部標準差的 5% 百分位數
    noise_proxy = np.percentile(sigma, 5) 
    
    # 設置一個最小值，防止極端情況下 h 接近 0
    return max(5.0, noise_proxy)