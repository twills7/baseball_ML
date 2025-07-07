import datetime
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def download_and_rename(positions="batter", year=2025, min_pa=10):
    # Setup download directory
    download_dir = os.path.abspath("data_snapshots")
    os.makedirs(download_dir, exist_ok=True)

    # Chrome options
    chrome_options = Options()
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--headless=new")

    # Start browser
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f"https://baseballsavant.mlb.com/leaderboard/custom?year={year}&type={positions}&min={min_pa}")

    # Wait and click Download CSV
    try:
        download_button = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.ID, "btnCSV"))
        )
        download_button.click()
        print("[Savant] Clicked Download CSV")
        time.sleep(5)  # Wait for download to finish

    finally:
        driver.quit()

    # Rename the downloaded file
    default_filename = os.path.join(download_dir, "stats.csv")
    today = datetime.date.today().isoformat()
    renamed_filename = os.path.join(download_dir, f"{positions}_{today}.csv")

    # Wait until the file actually appears (optional safety check)
    timeout = 10
    while timeout > 0 and not os.path.exists(default_filename):
        time.sleep(1)
        timeout -= 1

    if os.path.exists(renamed_filename):
        print(f"[Savant] File already exists: {renamed_filename}")
        return renamed_filename
    
    if os.path.exists(default_filename):
        os.rename(default_filename, renamed_filename)
        print(f"[Savant] Renamed to {renamed_filename}")
        return renamed_filename
    else:
        print("[Savant] Download failed or file not found.")
        return None

def csv_to_dataframe(file_path):
    import pandas as pd
    try:
        df = pd.read_csv(file_path)
        print(f"[Savant] Loaded {file_path} with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"[Savant] Error loading CSV: {e}")
        return None
    


if __name__ == "__main__":
    download_and_rename("batter")
    download_and_rename("pitcher")