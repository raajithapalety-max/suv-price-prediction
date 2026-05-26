from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import re

def scrape_all_spinny_cars():
    # Setup Chrome Options
    options = Options()
    # options.add_argument("--headless") # Keep commented out to watch the scrolling
    options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(options=options)
    base_url = "https://www.spinny.com/used-suv-cars-in-hyderabad/s/"
    
    print(f"1. Navigating to main page: {base_url}")
    driver.get(base_url)
    
    car_urls = []
    scraped_data = []
    
    try:
        # --- STEP 1: Infinite Scroll to load ALL cars ---
        print("\nScrolling to load all cars. Please wait...")
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "CarListingCardV2__carListingCardV2Root"))
        )
        
        previous_count = 0
        no_change_counter = 0
        
        while True:
            # Scroll to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3) # Wait for new cars
            
            # Count current links
            current_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/buy-used-cars/')]")
            current_count = len(current_links)
            
            if current_count == previous_count:
                no_change_counter += 1
                if no_change_counter >= 3: # If bottom reached
                    break
            else:
                print(f"  -> Loaded {current_count} cars so far...")
                no_change_counter = 0 
                
            previous_count = current_count

        # Extract unique URLs
        final_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/buy-used-cars/')]")
        for elem in final_links:
            url = elem.get_attribute('href')
            if url and url not in car_urls:
                car_urls.append(url)
                
        total_cars = len(car_urls)
        print(f"\n✅ Finished scrolling! Found {total_cars} unique car links.")
        print("Beginning detailed extraction. This will take ~20 minutes...\n")
        
        # --- STEP 2: Visit EVERY car's detail page ---
        for i, url in enumerate(car_urls, start=1):
            print(f"Scraping Car {i} / {total_cars}...")
            driver.get(url)
            
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
                time.sleep(2) # Buffer for elements to render
            except:
                print(f"  -> Page took too long to load, skipping...")
                continue
            
            # Initialize Dictionary (URL removed as requested)
            car_info = {
                "Name": "N/A",
                "Make_Year": "N/A",
                "Kilometers_Driven": "N/A",
                "Fuel_Type": "N/A",
                "Transmission": "N/A",
                "Owner_Count": "N/A",
                "Insurance_Validity": "N/A",
                "Price": "N/A",
                "Core_Systems_Score": "N/A",
                "Supporting_Systems_Score": "N/A",
                "Interiors_AC_Score": "N/A",
                "Exteriors_Lights_Score": "N/A",
                "Wear_Tear_Parts_Score": "N/A"
            }
            
            # 1. Get Car Name
            try:
                car_info["Name"] = driver.find_element(By.TAG_NAME, "h1").text.strip()
            except: pass
                
            # 2. Get Price
            try:
                price_elem = driver.find_element(By.XPATH, "//p[contains(@class, 'PriceSectionV3__ogPrice')]")
                car_info["Price"] = price_elem.text.strip()
            except:
                try:
                    fallback_price = driver.find_element(By.XPATH, "//*[contains(text(), '₹') and contains(text(), 'Lakh')]")
                    car_info["Price"] = fallback_price.text.strip()
                except: pass

            # 3. Get Overview Details
            try:
                overview_items = driver.find_elements(By.CLASS_NAME, "DesktopOverview__overviewItem")
                for item in overview_items:
                    try:
                        label = item.find_element(By.CLASS_NAME, "DesktopOverview__itemLabel").text.strip().lower()
                        value = item.find_element(By.CLASS_NAME, "DesktopOverview__itemDisplay").text.strip()
                        
                        if "make year" in label: car_info["Make_Year"] = value
                        elif "km driven" in label: car_info["Kilometers_Driven"] = value
                        elif "fuel type" in label: car_info["Fuel_Type"] = value
                        elif "transmission" in label: car_info["Transmission"] = value.split('\n')[0]
                        elif "owner" in label: car_info["Owner_Count"] = value
                        elif "insurance validity" in label: car_info["Insurance_Validity"] = value
                    except: continue 
            except: pass

            # 4. Get Quality Report Scores
            try:
                inspection_items = driver.find_elements(By.XPATH, "//*[contains(@class, 'InspectionReportV3__inspectionListV3')]/*")
                
                for item in inspection_items:
                    text_block = item.text.lower()
                    
                    score_match = re.search(r'(\d+\.\d+)', text_block)
                    score = score_match.group(1) if score_match else "N/A"
                    
                    if "core systems" in text_block:
                        car_info["Core_Systems_Score"] = score
                    elif "supporting systems" in text_block:
                        car_info["Supporting_Systems_Score"] = score
                    elif "interiors & ac" in text_block or "interiors" in text_block:
                        car_info["Interiors_AC_Score"] = score
                    elif "exteriors & lights" in text_block or "exteriors" in text_block:
                        car_info["Exteriors_Lights_Score"] = score
                    elif "wear & tear parts" in text_block or "wear & tear" in text_block:
                        car_info["Wear_Tear_Parts_Score"] = score
            except Exception as e:
                pass

            scraped_data.append(car_info)
            
            # Print a quick status to console
            if car_info['Name'] != "N/A":
                print(f"  -> Extracted: {car_info['Name'][:20]}... | {car_info['Price']} | Core Score: {car_info['Core_Systems_Score']}")

    except KeyboardInterrupt:
        print("\nScraping interrupted by user. Saving data collected so far...")
    except Exception as e:
        print(f"A critical error occurred: {e}")
        
    finally:
        driver.quit()
        print("\nBrowser closed.")
        
        # --- STEP 5: Save to CSV ---
        if scraped_data:
            df = pd.DataFrame(scraped_data)
            csv_filename = "spinny_full_dataset_final2.csv"
            df.to_csv(csv_filename, index=False)
            print(f"\n✅ SUCCESS! Dataset with {len(df)} cars saved to '{csv_filename}'")
        else:
            print("Failed to extract any data.")

if __name__ == "__main__":
    scrape_all_spinny_cars()