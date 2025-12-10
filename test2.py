import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from skimage import exposure, io
from skimage.exposure import match_histograms
from datetime import datetime
import warnings

# ปิด Warning
warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration
# ==========================================
ROOT_DIR = "Chest xray CP class"
OUTPUT_DIR = "AI_XRay_Report"
CLASSES = ["novap", "vap"]
CHANGE_THRESHOLD = 25

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. Enhancement Function (CLAHE)
# ==========================================
def enhance_contrast(image_arr: np.ndarray, clip_limit: float = 0.03):
    """ ปรับภาพให้ชัดด้วย CLAHE (Scikit-Image) """
    return exposure.equalize_adapthist(image_arr, clip_limit=clip_limit)

def apply_medical_enhancement(img_uint8):
    """ Wrapper แปลงภาพไป-กลับ 0-255 <-> 0-1 """
    img_float = img_uint8.astype(np.float32) / 255.0
    enhanced_float = enhance_contrast(img_float, clip_limit=0.03)
    return (enhanced_float * 255).astype(np.uint8)

# ==========================================
# 3. AI Helper: TorchXRayVision (Segmentation)
# ==========================================
# โหลดโมเดลรอไว้เลย (รันครั้งแรกจะโหลด Weight อัตโนมัติ)
print("Loading AI Model (PSPNet)...")
seg_model = xrv.baseline_models.chestx_det.PSPNet()
seg_model.eval() # Set to evaluation mode

def get_ai_lung_mask(img_numpy_uint8):
    """ 
    ใช้ TorchXRayVision ตัดปอดจากภาพ X-ray โดยเฉพาะ 
    (แก้ปัญหา lungmask ที่ใช้ไม่ได้กับ jpg)
    """
    try:
        # 1. เตรียมภาพให้เข้ากับ Model (ต้อง Normalize แบบเฉพาะของ xrv)
        # xrv ต้องการภาพช่วง -1024 ถึง 1024 และขนาด (1, 1, 512, 512)
        img_norm = xrv.datasets.normalize(img_numpy_uint8, 255) 
        
        # Resize เป็น 512x512 เพราะโมเดลนี้เทรนมาที่ความละเอียดนี้
        img_resized = cv2.resize(img_norm, (512, 512))
        
        # แปลงเป็น Tensor (Batch=1, Channel=1, H, W)
        img_tensor = torch.from_numpy(img_resized)[None, None, ...].float()
        
        # 2. รัน AI
        with torch.no_grad():
            outputs = seg_model(img_tensor)
        
        # 3. แปลงผลลัพธ์กลับเป็น Mask
        # Output ของ PSPNet จะมี 14 Channels (ปอดซ้าย, ขวา, หัวใจ, ฯลฯ)
        # Channel 4 = Left Lung, Channel 5 = Right Lung (Index อาจต่างกันเล็กน้อยตามเวอร์ชั่น แต่ปกตินี้คือปอด)
        # เราจะเอาทุก Channel ที่เป็น 'Lung' มารวมกัน
        pred = outputs[0].numpy()
        
        # targets ของโมเดล: ['Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
        # 'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', ...]
        # เราเอา Index 4 (Left Lung) และ 5 (Right Lung)
        mask_left = pred[4]
        mask_right = pred[5]
        
        # รวมปอดสองข้าง (ใช้ค่าความมั่นใจ > 0.5)
        combined_mask = np.maximum(mask_left, mask_right)
        mask_binary = np.where(combined_mask > 0.5, 255, 0).astype(np.uint8)
        
        # Resize Mask กลับไปเท่าภาพต้นฉบับ
        h, w = img_numpy_uint8.shape
        mask_final = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # ขยายขอบนิดนึง (Dilation)
        kernel = np.ones((5,5), np.uint8)
        mask_final = cv2.dilate(mask_final, kernel, iterations=1)
        
        return mask_final

    except Exception as e:
        print(f"   [Error] AI Segmentation failed: {e}")
        # Fallback: คืนค่าเป็นภาพดำ (ไม่ Mask) หรือใช้ Rectangle กลางภาพ
        return np.zeros_like(img_numpy_uint8)

# ==========================================
# 4. Registration & Analysis (เหมือนเดิม)
# ==========================================
def robust_registration(im_fixed, im_moving):
    # ใช้ภาพ Enhance จับจุด (แม่นยำกว่า)
    im_fixed_enh = apply_medical_enhancement(im_fixed)
    im_moving_enh = apply_medical_enhancement(im_moving)
    
    h, w = im_fixed.shape
    
    # SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im_fixed_enh, None)
    kp2, des2 = sift.detectAndCompute(im_moving_enh, None)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return im_moving, "Fail"

    # FLANN Matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            
    if len(good) > 10:
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        
        matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if matrix is not None:
            aligned = cv2.warpAffine(im_moving, matrix, (w, h))
            return aligned, "SIFT-Affine"
            
    # Fallback to ECC
    try:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        _, warp_matrix = cv2.findTransformECC(im_fixed_enh, im_moving_enh, warp_matrix, cv2.MOTION_AFFINE, criteria)
        aligned = cv2.warpAffine(im_moving, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned, "ECC"
    except:
        return im_moving, "Fail"

def analyze_clinical_change(img_path1, img_path2, patient_id, class_name, date1, date2, output_subfolder):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None: return None
    
    # 1. Resize & Enhance for display
    h, w = img1.shape
    img2 = cv2.resize(img2, (w, h))
    img1_show = apply_medical_enhancement(img1)
    
    # 2. Registration
    img2_aligned, method = robust_registration(img1, img2)
    print(f"   > Aligned {date1} vs {date2} via {method}")

    # 3. Histogram Match
    img2_matched = match_histograms(img2_aligned, img1).astype('uint8')
    img2_matched_show = apply_medical_enhancement(img2_matched)

    # 4. AI Segmentation (ใช้ฟังก์ชันใหม่)
    # ตัดปอดจากภาพ Baseline (Day 1) ที่ชัดที่สุด
    lung_mask = get_ai_lung_mask(img1)
    
    # Save Mask Check
    cv2.imwrite(os.path.join(output_subfolder, f"{patient_id}_{date1}_Mask.jpg"), lung_mask)

    # 5. Subtraction
    diff_worse = cv2.subtract(img2_matched, img1)
    diff_worse = cv2.bitwise_and(diff_worse, diff_worse, mask=lung_mask)
    _, mask_worse = cv2.threshold(diff_worse, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    diff_better = cv2.subtract(img1, img2_matched)
    diff_better = cv2.bitwise_and(diff_better, diff_better, mask=lung_mask)
    _, mask_better = cv2.threshold(diff_better, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Clean Noise
    kernel = np.ones((3,3), np.uint8)
    mask_worse = cv2.morphologyEx(mask_worse, cv2.MORPH_OPEN, kernel)
    mask_better = cv2.morphologyEx(mask_better, cv2.MORPH_OPEN, kernel)

    # 6. Calculate
    lung_px = cv2.countNonZero(lung_mask)
    if lung_px == 0: lung_px = 1
    
    worse_pct = (cv2.countNonZero(mask_worse) / lung_px) * 100
    better_pct = (cv2.countNonZero(mask_better) / lung_px) * 100
    
    if worse_pct > 3.0 and worse_pct > better_pct: status = "Worsening"
    elif better_pct > 3.0 and better_pct > worse_pct: status = "Improving"
    else: status = "Stable"

    # 7. Visualization
    result_visual = cv2.cvtColor(img2_matched_show, cv2.COLOR_GRAY2BGR)
    result_visual[mask_worse > 0] = [0, 0, 255] # แดง
    result_visual[(mask_better > 0) & (mask_worse == 0)] = [0, 255, 0] # เขียว
    
    # Draw Mask Outline
    contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_visual, contours, -1, (0, 255, 255), 2) # เหลือง

    final = cv2.addWeighted(cv2.cvtColor(img1_show, cv2.COLOR_GRAY2BGR), 0.5, result_visual, 0.5, 0)
    
    filename = f"{patient_id}_{date1}_vs_{date2}.jpg"
    save_path = os.path.join(output_subfolder, filename)
    cv2.imwrite(save_path, final)

    return {
        "Patient_ID": patient_id,
        "Dates": f"{date1} vs {date2}",
        "Status": status,
        "New_Lesion(%)": round(worse_pct, 2),
        "Improved(%)": round(better_pct, 2),
        "Method": method
    }

# ==========================================
# Helpers & Main
# ==========================================
def parse_info(filename):
    try:
        parts = filename.replace(".jpg", "").split("_")
        return datetime.strptime(parts[-2], "%Y%m%d"), int(parts[-1])
    except: return None, None

def get_valid_image_sequence(folder_path):
    if not os.path.exists(folder_path): return []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    daily = {}
    for f in files:
        d, s = parse_info(f)
        if d and (d not in daily or s > daily[d][0]): daily[d] = (s, f)
    return [daily[d][1] for d in sorted(daily.keys())]

def main():
    report = []
    print(f"Running X-Ray Specific AI Analysis... Output: {OUTPUT_DIR}")
    
    for cls in CLASSES:
        c_path = os.path.join(ROOT_DIR, cls)
        if not os.path.exists(c_path): continue
        
        for pid in os.listdir(c_path):
            p_path = os.path.join(c_path, pid)
            if not os.path.isdir(p_path): continue
            
            files = get_valid_image_sequence(p_path)
            if len(files) < 2: continue
            
            save_dir = os.path.join(OUTPUT_DIR, cls, pid)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            
            print(f"Processing Patient: {pid}")
            
            for i in range(len(files)-1):
                f1, f2 = files[i], files[i+1]
                d1, _ = parse_info(f1)
                d2, _ = parse_info(f2)
                
                res = analyze_clinical_change(
                    os.path.join(p_path, f1), 
                    os.path.join(p_path, f2),
                    pid, cls, d1.date(), d2.date(), save_dir
                )
                if res: report.append(res)

    if report:
        pd.DataFrame(report).to_excel(os.path.join(OUTPUT_DIR, "AI_Report.xlsx"), index=False)
        print("Done.")

if __name__ == "__main__":
    main()