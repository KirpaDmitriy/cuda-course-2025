import os
import requests
from concurrent.futures import ThreadPoolExecutor

COCO_URLS_FILE = "http://images.cocodataset.org/zips/val2017.zip"

SAVE_DIR = "../data/calibration_images"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_image(i):
    # Формат URL COCO: http://images.cocodataset.org/val2017/000000xxxxxx.jpg
    image_id = str(i).zfill(12) 
    url = f"http://images.cocodataset.org/val2017/{image_id}.jpg"
    
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(os.path.join(SAVE_DIR, f"{image_id}.jpg"), 'wb') as f:
                f.write(r.content)
            print(f"Downloaded {image_id}.jpg")
    except Exception as e:
        print(f"Error {i}: {e}")

print("Start downloading calibration images...")
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(download_image, [39769, 37777, 785, 1000, 2000, 3000, 4000, 5000, 6000, 7000] + list(range(139, 2000, 10)))

print(f"Загрузка завершена. Файлы в {SAVE_DIR}")
