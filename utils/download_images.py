import os
import csv
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 50  

def check_internet():
    try:
        requests.get('https://www.google.com', timeout=5)
        return True
    except:
        return False
        

def download_image_with_retry(img_url, save_path, idx, total):
    while True:
        if check_internet():
            try:
                response = requests.get(img_url, timeout=20)
                if response.status_code == 200:
                    with open(save_path, "wb") as f:  # This always overwrites any old file
                        f.write(response.content)
                    print(f"\rDownloaded {idx+1}/{total}: {os.path.basename(save_path)}", end="")
                    return
                else:
                    print(f"\nFailed ({response.status_code}): {img_url}, retrying in 10s.", end="")
            except Exception as e:
                print(f"\nError downloading {img_url}: {e}, retrying in 10s.", end="")
        else:
            print("\nNo internet connection! Retrying in 10s.", end="")
        time.sleep(10)

def process_csv_parallel(csv_file):
    folder_name = os.path.splitext(os.path.basename(csv_file))[0]
    os.makedirs(folder_name, exist_ok=True)
    with open(csv_file, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
        total = len(rows)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for idx, row in enumerate(rows):
                img_url = row['image_link'].strip()
                sample_id = row['sample_id'].strip()
                if img_url and img_url.startswith('http') and sample_id:
                    ext = os.path.splitext(img_url)[1]
                    if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                        ext = '.jpg'
                    save_path = os.path.join(folder_name, f"{sample_id}{ext}")
                    futures.append(executor.submit(
                        download_image_with_retry, img_url, save_path, idx, total
                    ))
            for future in as_completed(futures):
                pass
    print("\nAll downloads complete.")

if _name_ == "_main_":
    process_csv_parallel("train.csv")
