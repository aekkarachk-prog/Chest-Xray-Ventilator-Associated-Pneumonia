import os
import cv2
import numpy as np
import pandas as pd
from skimage.exposure import match_histograms
from datetime import datetime

# ==========================================
# 1. การตั้งค่า (Configuration)
# ==========================================
ROOT_DIR = "Chest xray CP class"
OUTPUT_DIR = "Clinical_Report_Output"
CLASSES = ["novap", "vap"]

# ค่าความไวในการจับความเปลี่ยนแปลง (Threshold)
# ค่าต่ำ = จับละเอียด (อาจติด Noise), ค่าสูง = จับเฉพาะจุดชัดๆ
CHANGE_THRESHOLD = 25 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. ฟังก์ชันจัดการไฟล์
# ==========================================
def parse_info(filename):
    try:
        parts = filename.replace(".jpg", "").split("_")
        date_str = parts[-2]
        seq = int(parts[-1])
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        return date_obj, seq
    except Exception:
        return None, None

def get_valid_image_sequence(folder_path):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    daily_images = {}
    for f in files:
        date_obj, seq = parse_info(f)
        if date_obj is None: continue
        if date_obj not in daily_images:
            daily_images[date_obj] = (seq, f)
        else:
            if seq > daily_images[date_obj][0]:
                daily_images[date_obj] = (seq, f)
    sorted_items = sorted(daily_images.items(), key=lambda x: x[0])
    return [item[1][1] for item in sorted_items]

# ==========================================
# 3. ฟังก์ชันวิเคราะห์ รอยโรค (Core Logic)
# ==========================================
def analyze_clinical_change(img_path1, img_path2, patient_id, class_name, date1, date2, output_subfolder):
    # อ่านภาพ
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None: return None

    # 1. Resize & Align
    h, w = img1.shape
    img2 = cv2.resize(img2, (w, h))
    
    # 2. Histogram Matching (สำคัญมาก: ปรับแสงภาพ 2 ให้เท่าภาพ 1)
    img2_matched = match_histograms(img2, img1).astype('uint8')

    # =========================================================
    # ส่วนสำคัญ: แยกแยะ "แย่ลง" (ขาวขึ้น) vs "ดีขึ้น" (ดำลง)
    # =========================================================
    
    # A. หาพื้นที่ที่ "ขาวขึ้น" (Worsening / New Infiltration)
    # ตรรกะ: (Day 2 - Day 1) > 0
    diff_worse = cv2.subtract(img2_matched, img1)
    _, mask_worse = cv2.threshold(diff_worse, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # B. หาพื้นที่ที่ "ดำลง" (Improvement)
    # ตรรกะ: (Day 1 - Day 2) > 0
    diff_better = cv2.subtract(img1, img2_matched)
    _, mask_better = cv2.threshold(diff_better, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)

    # C. กรอง Noise (จุดเล็กๆ น้อยๆ ไม่นับ)
    kernel = np.ones((3,3), np.uint8)
    mask_worse = cv2.morphologyEx(mask_worse, cv2.MORPH_OPEN, kernel)
    mask_better = cv2.morphologyEx(mask_better, cv2.MORPH_OPEN, kernel)

    # =========================================================
    # 4. คำนวณเปอร์เซ็นต์ (Quantification)
    # =========================================================
    total_pixels = h * w
    worse_pixels = cv2.countNonZero(mask_worse)
    better_pixels = cv2.countNonZero(mask_better)

    worse_percent = (worse_pixels / total_pixels) * 100
    better_percent = (better_pixels / total_pixels) * 100
    
    # Net Change: ถ้าบวกแปลว่าแย่ลงรวมๆ, ถ้าลบแปลว่าดีขึ้นรวมๆ
    net_change = worse_percent - better_percent 

    # สรุปผล (Diagnosis Logic)
    if worse_percent > 0.5 and worse_percent > better_percent:
        status = "Worsening (New Lesion)"
    elif better_percent > 0.5 and better_percent > worse_percent:
        status = "Improving"
    else:
        status = "Stable / No Significant Change"

    # =========================================================
    # 5. สร้างภาพผลลัพธ์ (Visualization)
    # =========================================================
    # สร้างภาพสีเพื่อวาด Heatmap
    result_visual = cv2.cvtColor(img2_matched, cv2.COLOR_GRAY2BGR)
    
    # สีแดง = แย่ลง (New Infiltration) -> (0, 0, 255)
    # สีเขียว = ดีขึ้น (Improvement) -> (0, 255, 0)
    
    # ไฮไลท์จุดที่แย่ลง (สีแดง)
    result_visual[mask_worse > 0] = [0, 0, 200] 
    
    # ไฮไลท์จุดที่ดีขึ้น (สีเขียว) - วาดทับเฉพาะส่วนที่ไม่ชนกับสีแดง
    result_visual[(mask_better > 0) & (mask_worse == 0)] = [0, 200, 0]

    # ผสมภาพกับต้นฉบับให้ดูโปร่งใส (Alpha Blending)
    img2_color = cv2.cvtColor(img2_matched, cv2.COLOR_GRAY2BGR)
    final_overlay = cv2.addWeighted(img2_color, 0.6, result_visual, 0.4, 0)

    # Save
    filename_base = f"{patient_id}_{date1}_vs_{date2}"
    save_path = os.path.join(output_subfolder, f"{filename_base}_ClinicalMap.jpg")
    cv2.imwrite(save_path, final_overlay)

    return {
        "Patient_ID": patient_id,
        "Group": class_name,
        "Compare": f"{date1} vs {date2}",
        "New_Lesion_Area (%)": round(worse_percent, 2),
        "Improvement_Area (%)": round(better_percent, 2),
        "Net_Status": status,
        "Image_Path": save_path
    }

# ==========================================
# Main Execution
# ==========================================
def main():
    all_reports = []
    print(f"Starting Clinical Analysis... Saving to '{OUTPUT_DIR}'")

    for cls in CLASSES:
        class_path = os.path.join(ROOT_DIR, cls)
        if not os.path.exists(class_path): continue

        patient_ids = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
        
        for pid in patient_ids:
            patient_folder = os.path.join(class_path, pid)
            sorted_files = get_valid_image_sequence(patient_folder)
            
            if len(sorted_files) < 2: continue

            save_path = os.path.join(OUTPUT_DIR, cls, pid)
            if not os.path.exists(save_path): os.makedirs(save_path)

            print(f"Analyzing Patient: {pid}...")

            for i in range(len(sorted_files) - 1):
                file1 = sorted_files[i]
                file2 = sorted_files[i+1]
                
                d1 = parse_info(file1)[0].strftime("%Y-%m-%d")
                d2 = parse_info(file2)[0].strftime("%Y-%m-%d")
                
                data = analyze_clinical_change(
                    os.path.join(patient_folder, file1),
                    os.path.join(patient_folder, file2),
                    pid, cls, d1, d2, save_path
                )
                if data:
                    all_reports.append(data)

    # Save Excel Report
    if all_reports:
        df = pd.DataFrame(all_reports)
        excel_path = os.path.join(OUTPUT_DIR, "Clinical_Finding_Report.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\nReport Generated: {excel_path}")
        print(df.head())
    else:
        print("No valid image pairs found.")

if __name__ == "__main__":
    main()