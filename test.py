import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms
from datetime import datetime

# ==========================================
# ส่วนตั้งค่า (Configuration)
# ==========================================
ROOT_DIR = "Chest xray CP class"  # โฟล์เดอร์หลักของคุณ
OUTPUT_DIR = "Comparison_Results" # โฟล์เดอร์ที่จะเก็บผลลัพธ์
CLASSES = ["novap", "vap"]        # คลาสที่ต้องการทำ

# สร้างโฟล์เดอร์ผลลัพธ์ถ้ายังไม่มี
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. ฟังก์ชันจัดการชื่อไฟล์และคัดเลือก
# ==========================================
def parse_info(filename):
    """
    แยกข้อมูลจากชื่อไฟล์: 2131_cxr_vap_20250808_1.jpg
    Return: date_obj, sequence, full_path
    """
    try:
        parts = filename.replace(".jpg", "").split("_")
        # parts structure: [ID, 'cxr', class, YYYYMMDD, Seq]
        # ตัวอย่าง: parts[-2] คือวันที่, parts[-1] คือลำดับ
        date_str = parts[-2]
        seq = int(parts[-1])
        
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        return date_obj, seq
    except Exception as e:
        return None, None

def get_valid_image_sequence(folder_path):
    """
    อ่านไฟล์ทั้งหมดในโฟล์เดอร์ และใช้กฎ: วันเดียวกันเอา Sequence สูงสุด
    Return: List ของไฟล์ภาพที่เรียงตามวันที่แล้ว
    """
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    # Dictionary เก็บ { date_obj: (max_seq, filename) }
    daily_images = {}

    for f in files:
        date_obj, seq = parse_info(f)
        if date_obj is None: continue # ข้ามไฟล์ที่ชื่อผิดรูปแบบ

        # ตรรกะการคัดเลือก (Selection Logic)
        if date_obj not in daily_images:
            daily_images[date_obj] = (seq, f)
        else:
            # ถ้ามีวันนี้อยู่แล้ว เช็คว่าไฟล์ใหม่ seq สูงกว่าไหม?
            current_max_seq, _ = daily_images[date_obj]
            if seq > current_max_seq:
                daily_images[date_obj] = (seq, f) # แทนที่ด้วยตัวใหม่ (เช่น _2 แทน _1)

    # แปลงกลับเป็น List และเรียงตามวันที่
    # sorted_items จะเป็น list ของ tuples: (date, (seq, filename))
    sorted_items = sorted(daily_images.items(), key=lambda x: x[0])
    
    # ดึงเฉพาะชื่อไฟล์ออกมา
    sorted_filenames = [item[1][1] for item in sorted_items]
    return sorted_filenames

# ==========================================
# 2. ฟังก์ชันประมวลผลและเปรียบเทียบ
# ==========================================
def compare_images(img_path1, img_path2, patient_id, date1, date2, output_subfolder):
    # อ่านภาพเป็น Grayscale
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"Error reading images for {patient_id}")
        return

    # --- Preprocessing: Resize ---
    # บังคับให้ภาพที่ 2 มีขนาดเท่ากับภาพที่ 1 เพื่อให้เปรียบเทียบ pixel ต่อ pixel ได้
    h, w = img1.shape
    img2_resized = cv2.resize(img2, (w, h))

    # --- Preprocessing: Histogram Matching ---
    # ปรับแสงของภาพที่ 2 ให้ใกล้เคียงภาพที่ 1 (ลดผลกระทบจากการถ่ายคนละแสง)
    img2_matched = match_histograms(img2_resized, img1)
    img2_matched = img2_matched.astype('uint8')

    # --- 1. Calculate SSIM (Similarity) ---
    score, diff = ssim(img1, img2_matched, full=True)
    diff = (diff * 255).astype("uint8")
    similarity_percent = score * 100

    # --- 2. Calculate Absolute Difference (Subtraction) ---
    # ใช้ absdiff เพื่อดูความต่างแบบดิบๆ
    abs_diff = cv2.absdiff(img1, img2_matched)
    
    # เพิ่ม Contrast ให้ภาพ Difference เห็นชัดขึ้น (Optional)
    _, thresh_diff = cv2.threshold(abs_diff, 25, 255, cv2.THRESH_BINARY)

    # ==========================================
    # 3. สร้าง Output ทั้ง 3 แบบ
    # ==========================================
    
    filename_base = f"{patient_id}_{date1}_vs_{date2}"
    
    # Output 1: Side-by-Side (ภาพจริงเทียบกัน)
    combined_visual = np.hstack((img1, img2_matched))
    cv2.imwrite(os.path.join(output_subfolder, f"{filename_base}_SideBySide.jpg"), combined_visual)

    # Output 2: Subtraction/Difference Map (เน้นจุดต่าง)
    # เราจะเซฟภาพ Difference ดิบ และภาพที่ทำสี Heatmap เพื่อให้ดูง่าย
    heatmap_img = cv2.applyColorMap(abs_diff, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_subfolder, f"{filename_base}_DiffMap.jpg"), heatmap_img)

    # Output 3: Similarity Info (และภาพที่ SSIM ไฮไลท์ความต่าง)
    # เขียน text ลงบนภาพ SSIM Diff
    cv2.putText(diff, f"Similarity: {similarity_percent:.2f}%", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    cv2.imwrite(os.path.join(output_subfolder, f"{filename_base}_SSIM.jpg"), diff)

    print(f"   [Process] Compared {date1} vs {date2} | Similarity: {similarity_percent:.2f}%")

# ==========================================
# Main Execution Loop
# ==========================================
def main():
    for cls in CLASSES:
        class_path = os.path.join(ROOT_DIR, cls)
        if not os.path.exists(class_path):
            print(f"Folder not found: {class_path}")
            continue

        # วนลูปรายคน (Patient ID)
        patient_ids = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
        
        for pid in patient_ids:
            patient_folder = os.path.join(class_path, pid)
            print(f"\nProcessing Patient: {pid} (Class: {cls})")
            
            # 1. คัดเลือกและเรียงลำดับไฟล์
            sorted_files = get_valid_image_sequence(patient_folder)
            
            if len(sorted_files) < 2:
                print(f"   Not enough images to compare for {pid}")
                continue

            # สร้างโฟล์เดอร์เก็บผลลัพธ์แยกรายคน
            save_path = os.path.join(OUTPUT_DIR, cls, pid)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 2. จับคู่เปรียบเทียบ (Time Series Loop)
            # เปรียบเทียบ Day 0 vs Day 1, Day 1 vs Day 2, ...
            for i in range(len(sorted_files) - 1):
                file1 = sorted_files[i]
                file2 = sorted_files[i+1]
                
                # ดึงวันที่มาตั้งชื่อไฟล์ผลลัพธ์
                date1 = file1.split("_")[-2]
                date2 = file2.split("_")[-2]
                
                path1 = os.path.join(patient_folder, file1)
                path2 = os.path.join(patient_folder, file2)
                
                compare_images(path1, path2, pid, date1, date2, save_path)

if __name__ == "__main__":
    main()