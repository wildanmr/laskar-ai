from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import random
import time

chrome_options = Options()
chrome_options.add_argument("--headless=new")  # Enables headless mode
chrome_options.add_argument("--disable-gpu")  # Disables GPU hardware acceleration
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

service = Service(executable_path='/bin/chromedriver')

driver = webdriver.Chrome(service=service, options=chrome_options)

url = 'https://steamcommunity.com/app/2246340/positivereviews/?browsefilter=toprated&snr=1_5_100010_&filterLanguage=english'

page = 1
reviews = []
count = 0

driver.get(url)
while True:
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    page_div = soup.select_one(f"div#page{page}")
    
    if not page_div:
        break
        
    review_cards = page_div.select("div.apphub_Card.modalContentLink.interactable")

    for review_card in review_cards:
        card = {
            "url": review_card.get("data-modal-content-url"),
            "review": "",
            "is_recommended": ""
        }
        
        text_container = review_card.select_one("div.apphub_CardTextContent")
        if text_container:
            date_div = text_container.select_one("div.date_posted")
            if date_div:
                date_div.extract()
            
            for br in text_container.find_all('br'):
                br.replace_with('\n')
            
            card["review"] = text_container.get_text(strip=True)
        
        rec_element = review_card.select_one("div.reviewInfo div.title")
        if rec_element:
            card["is_recommended"] = rec_element.text
        
        reviews.append(card)

    print(f"Collected: {len(reviews)} reviews")
    page += 1

    if len(reviews) % 50 == 0:
        pd.DataFrame(reviews).to_csv('mhwilds-reviews-3.csv', mode='a', header=False, index=False)
        count += len(reviews)
        print(f"Total {count} reviews, saved")
        reviews.clear()

    delay = random.uniform(2, 5)
    time.sleep(delay)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, f"div#page{page}"))
        )
    except TimeoutException:
        print(f"No more pages after page {page-1}. Exiting.")
        break