"""
Using undetected-chromedriver to bypass bot detection
This is often the most effective method for heavily protected sites
"""

try:
    import undetected_chromedriver as uc
    UNDETECTED_AVAILABLE = True
except ImportError:
    print("⚠️ undetected-chromedriver not available. Install with: pip install undetected-chromedriver")
    UNDETECTED_AVAILABLE = False
    from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

def setup_undetected_driver():
    """Setup undetected Chrome driver"""
    if not UNDETECTED_AVAILABLE:
        print("❌ Undetected chromedriver not available, falling back to regular selenium")
        return setup_regular_driver()
    
    try:
        options = uc.ChromeOptions()
        
        # Add some options but keep it minimal for undetected-chromedriver
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        
        # Create undetected Chrome driver
        driver = uc.Chrome(options=options, version_main=None)
        
        # Maximize window
        driver.maximize_window()
        
        print("✅ Undetected Chrome driver initialized successfully")
        return driver
        
    except Exception as e:
        print(f"❌ Failed to initialize undetected driver: {e}")
        print("🔄 Falling back to regular selenium...")
        return setup_regular_driver()

def setup_regular_driver():
    """Fallback to regular selenium driver"""
    from selenium import webdriver
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.maximize_window()
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    print("✅ Regular Chrome driver initialized")
    return driver

def wait_for_human_interaction(driver, message="Please solve any CAPTCHA or verification manually, then press Enter to continue...", auto_continue=False):
    """Wait for human to solve CAPTCHA or other challenges"""
    print(f"🤖 {message}")
    if auto_continue:
        print("🔄 Auto-continuing without waiting for input...")
        time.sleep(2)  # Brief pause for visibility
    else:
        input("Press Enter when ready to continue...")
    print("✅ Continuing...")

def auto_continue(message="Automatically continuing to next page..."):
    """Automatically continue without waiting for user input"""
    print(f"🔄 {message}")
    time.sleep(random.uniform(2, 4))  # Random delay to appear more natural
    print("✅ Continuing...")

def extract_jobs_from_page(driver, page_url):
    """Extract jobs from a single page using exact selectors from original code"""
    try:
        print(f"🔗 Loading: {page_url}")
        driver.get(page_url)

        # Wait for page to load
        time.sleep(random.uniform(3, 6))

        # Check if we need human intervention (CAPTCHA, etc.)
        page_source = driver.page_source.lower()
        if any(keyword in page_source for keyword in ['captcha', 'verification', 'challenge', 'blocked']):
            print("🚫 Detected possible CAPTCHA or verification challenge")
            wait_for_human_interaction(driver)

        # Wait for job listings to appear using exact selector from original code
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".job-card"))
            )
        except:
            print("⚠️ No job listings found with .job-card selector")
        
        # Extract job listings using exact selector from original code
        job_elements = driver.find_elements(By.CSS_SELECTOR, ".job-card")
        print(f"📋 Found {len(job_elements)} job listings on this page")

        if len(job_elements) == 0:
            print("⚠️ No job listings found. Possible CAPTCHA or page structure changed.")
            return []

        jobs_data = []
        
        # First, collect all job URLs and basic info
        job_info_list = []
        for i, job_element in enumerate(job_elements, 1):
            try:
                # Extract basic info
                title = job_element.find_element(By.CSS_SELECTOR, "h3.imt-3.text-break").text.strip()
                company = job_element.find_element(By.CSS_SELECTOR, "span.ims-2 a[href*='/companies/']").text.strip()
                location = job_element.find_element(By.CSS_SELECTOR, "div.text-rich-grey.text-truncate").text.strip()

                # Get job URL
                try:
                    job_link = job_element.find_element(By.CSS_SELECTOR, "h3 a, a[href*='/it-jobs/']")
                    job_url = job_link.get_attribute('href')
                except:
                    job_url = None

                job_info_list.append({
                    'id': i,
                    'title': title,
                    'company': company,
                    'location': location,
                    'url': job_url
                })
                print(f"📝 Collected basic info for job {i}: {title} at {company}")
            except Exception as e:
                print(f"❌ Error collecting basic info for job {i}: {e}")
                continue

        print(f"📋 Collected {len(job_info_list)} job URLs. Now processing details...")

        # Now process each job URL for detailed information
        for job_info in job_info_list:
            # Initialize default values for each job to avoid carrying over from previous job
            salary = "Not specified"
            work_type = "Not specified"
            skills = []
            description = "No description available"
            requirements = "No requirements specified"

            try:
                i = job_info['id']
                title = job_info['title']
                company = job_info['company']
                location = job_info['location']
                job_url = job_info['url']

                print(f"🔍 Processing detailed info for job {i}: {title}")

                if not job_url:
                    print(f"⚠️ No URL found for job {i}, skipping detailed extraction")
                    # Create job data with basic info only
                    job_data = {
                        'id': i,
                        'title': title,
                        'company': company,
                        'location': location,
                        'salary': salary,
                        'work_type': work_type,
                        'description': description,
                        'requirements': requirements,
                        'skills': ", ".join(skills) if skills else "Not specified"
                    }
                    jobs_data.append(job_data)
                    continue

                # Navigate to job detail page
                print(f"🔗 Navigating to: {job_url}")
                driver.get(job_url)

                # Wait for job details page to load
                time.sleep(random.uniform(3, 5))

                # Wait for job details to load in right panel and ensure it's for the current job
                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".job-description, .preview-job-wrapper"))
                    )
                    print(f"📄 Job details panel loaded for job {i}")

                    # Additional wait to ensure content is fully loaded
                    time.sleep(2)

                    # Verify that the detail panel shows the correct job by checking title
                    try:
                        detail_title_elements = driver.find_elements(By.CSS_SELECTOR, "h1, h2, .job-title")
                        detail_title = ""
                        for elem in detail_title_elements:
                            if elem.is_displayed() and elem.text.strip():
                                detail_title = elem.text.strip()
                                break

                        if detail_title and title.lower() in detail_title.lower():
                            print(f"✅ Detail panel shows correct job: {detail_title}")
                        else:
                            print(f"⚠️ Detail panel title mismatch. Expected: {title}, Got: {detail_title}")
                    except:
                        print(f"⚠️ Could not verify detail panel title for job {i}")

                except Exception as e:
                    print(f"⚠️ Job details panel failed to load for job {i}: {e}")
                    continue

                # Extract salary using updated selector based on actual HTML structure
                try:
                    salary_element = driver.find_element(By.CSS_SELECTOR, ".salary.text-success-color span.ips-2.fw-500")
                    salary = salary_element.text.strip()
                    print(f"💰 Found salary: {salary}")
                except Exception as e:
                    print(f"No salary found for job {i}: {e}")
                    salary = "Not specified"

                # Extract work type using updated selector based on actual HTML structure
                try:
                    work_type_element = driver.find_element(By.CSS_SELECTOR, "span.normal-text.text-rich-grey.ms-1")
                    work_type = work_type_element.text.strip()
                    print(f"💼 Found work type: {work_type}")
                except Exception as e:
                    print(f"No work type found for job {i}: {e}")
                    work_type = "Not specified"

                # Extract skills from multiple sections using exact XPath from original code
                skills = []
                try:
                    # Skills section - using XPath for better text matching
                    skill_elements = driver.find_elements(By.XPATH, "//div[contains(text(), 'Skills:')]/following-sibling::div//a[@class='text-reset itag itag-light itag-sm']")
                    skills.extend([skill.text.strip() for skill in skill_elements])

                    # Job Expertise section
                    expertise_elements = driver.find_elements(By.XPATH, "//div[contains(text(), 'Job Expertise:')]/following-sibling::div//a[@class='text-reset itag itag-light itag-sm']")
                    skills.extend([skill.text.strip() for skill in expertise_elements])

                    # Job Domain section
                    domain_elements = driver.find_elements(By.XPATH, "//div[contains(text(), 'Job Domain:')]/following-sibling::div//div[@class='itag bg-light-grey itag-sm cursor-default']")
                    skills.extend([skill.text.strip() for skill in domain_elements])

                    print(f"🔧 Found {len(skills)} skills: {skills[:5]}..." if len(skills) > 5 else f"🔧 Found skills: {skills}")
                except Exception as e:
                    print(f"Error extracting skills for job {i}: {e}")
                    skills = []

                # Extract description using updated selector based on actual HTML structure
                try:
                    # Look for Job description section
                    description_element = driver.find_element(By.XPATH, "//h2[contains(text(), 'Job description')]/following-sibling::*")
                    description = description_element.text.strip()
                    print(f"📝 Found description: {description[:50]}...")
                except Exception as e:
                    # Fallback to general paragraph selector
                    try:
                        description_element = driver.find_element(By.CSS_SELECTOR, ".imy-5.paragraph")
                        description = description_element.text.strip()
                        print(f"📝 Found description (fallback): {description[:50]}...")
                    except:
                        print(f"No description found for job {i}: {e}")
                        description = "No description available"

                # Extract requirements/experience using updated selector based on actual HTML structure
                try:
                    # Look for "Your skills and experience" section
                    requirements_element = driver.find_element(By.XPATH, "//h2[contains(text(), 'Your skills and experience')]/following-sibling::*")
                    requirements = requirements_element.text.strip()
                    print(f"📋 Found requirements: {requirements[:50]}...")
                except Exception as e:
                    print(f"No requirements found for job {i}: {e}")
                    requirements = "No requirements specified"

                job_data = {
                    'id': i,
                    'title': title,
                    'company': company,
                    'location': location,
                    'salary': salary,
                    'work_type': work_type,
                    'description': description,
                    'requirements': requirements,
                    'skills': ", ".join(skills) if skills else "Not specified"
                }

                # Debug: Print extracted data for verification
                print(f"✅ Extracted job {i}:")
                print(f"   Title: {title}")
                print(f"   Company: {company}")
                print(f"   Salary: {salary}")
                print(f"   Work Type: {work_type}")
                print(f"   Skills: {', '.join(skills) if skills else 'Not specified'}")
                print(f"   Description: {description[:100]}..." if len(description) > 100 else f"   Description: {description}")
                print(f"   Requirements: {requirements[:100]}..." if len(requirements) > 100 else f"   Requirements: {requirements}")
                print("---")

                jobs_data.append(job_data)

                # Small delay between jobs
                time.sleep(random.uniform(1, 2))

            except Exception as e:
                print(f"❌ Error extracting job details: {e}")
                continue
        
        return jobs_data
        
    except Exception as e:
        print(f"❌ Error processing page {page_url}: {e}")
        return []

def crawl_itviec_undetected(num_pages=3):
    """Main crawling function using undetected chromedriver"""
    driver = setup_undetected_driver()
    all_jobs_data = []
    
    try:
        # Start with homepage to establish session
        print("🏠 Loading homepage first...")
        driver.get("https://itviec.com/")
        time.sleep(random.uniform(3, 6))
        
        # Check for initial challenges
        if any(keyword in driver.page_source.lower() for keyword in ['captcha', 'verification', 'challenge']):
            wait_for_human_interaction(driver, "Please solve any initial verification, then press Enter...")
        
        for page in range(1, num_pages + 1):
            print(f"\n📄 Processing page {page}/{num_pages}...")

            if page == 1:
                page_url = "https://itviec.com/it-jobs"
            else:
                page_url = f"https://itviec.com/it-jobs?page={page}"

            jobs_data = extract_jobs_from_page(driver, page_url)
            all_jobs_data.extend(jobs_data)

            print(f"✅ Page {page} completed. Total jobs collected: {len(all_jobs_data)}")

            # Auto continue to next page without manual input
            if page < num_pages:
                auto_continue(f"Moving to page {page + 1}/{num_pages}...")
                delay = random.uniform(5, 10)
                print(f"⏳ Additional delay: {delay:.1f}s before next page...")
                time.sleep(delay)
        
    except Exception as e:
        print(f"💥 Critical error: {e}")
    finally:
        print("🔚 Closing browser...")
        try:
            driver.quit()
        except Exception as e:
            print(f"⚠️ Error closing driver (this is normal): {e}")
            try:
                driver.close()
            except:
                pass
    
    return all_jobs_data

def save_to_csv(data, filename="itviec_jobs_undetected.csv"):
    """Save data to CSV file"""
    if not data:
        print("⚠️ No data to save")
        return
    
    df = pd.DataFrame(data)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"💾 Data saved to {output_path}")

if __name__ == "__main__":
    print("🚀 Starting ITViec crawler with undetected-chromedriver...")
    print("💡 This method is most effective against bot detection")
    print("🤖 You may need to solve CAPTCHA manually when prompted")
    
    jobs = crawl_itviec_undetected(num_pages=51)
    save_to_csv(jobs)
    print(f"🎉 Crawling completed. Collected {len(jobs)} jobs.")
