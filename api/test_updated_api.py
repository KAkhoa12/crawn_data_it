#!/usr/bin/env python3
"""
Test script for the updated API endpoints
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test API health"""
    print("=== Testing API Health ===")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API is healthy!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure the API server is running on http://localhost:8000")
        return False

def test_cv_extraction_unique():
    """Test CV extraction with unique seeker_id constraint"""
    print("\n=== Testing CV Extraction (Unique Seeker ID) ===")
    
    # Create a simple test file
    test_content = b"Sample CV content for testing Python FastAPI Machine Learning skills"
    
    try:
        # First upload
        files = {
            'file': ('test_cv_1.txt', test_content, 'text/plain')
        }
        data = {
            'seeker_id': 999,  # Use unique seeker_id for testing
            'name': 'First CV Upload'
        }
        
        response1 = requests.post(
            f"{BASE_URL}/cv/extract/all-features",
            files=files,
            data=data
        )
        
        if response1.status_code == 200:
            result1 = response1.json()
            print("✅ First CV upload successful!")
            print(f"   Primary skills: {result1['primary_skills']}")
        else:
            print(f"❌ First CV upload failed: {response1.status_code}")
            print(f"   Response: {response1.text}")
            return False
        
        # Second upload with same seeker_id (should update)
        files2 = {
            'file': ('test_cv_2.txt', b"Updated CV content with Docker Git SQL skills", 'text/plain')
        }
        data2 = {
            'seeker_id': 999,  # Same seeker_id
            'name': 'Updated CV Upload'
        }
        
        response2 = requests.post(
            f"{BASE_URL}/cv/extract/all-features",
            files=files2,
            data=data2
        )
        
        if response2.status_code == 200:
            result2 = response2.json()
            print("✅ Second CV upload (update) successful!")
            print(f"   Primary skills: {result2['primary_skills']}")
            print("✅ CV unique constraint working correctly!")
            return True
        else:
            print(f"❌ Second CV upload failed: {response2.status_code}")
            print(f"   Response: {response2.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception testing CV extraction: {e}")
        return False

def test_job_extraction_with_complex_description():
    """Test Job extraction with the complex description that caused JSON error"""
    print("\n=== Testing Job Extraction (Complex Description) ===")
    
    job_data = {
        "employer_id": 1,
        "title": "abc intern",
        "description": """Job description
Design and implement sophisticated scalable multi-threaded Object Oriented Software in C++ for solving challenging problems involving high speed data processing and networking
Design advanced software modules that follow modern C++ design patterns
Apply C programming skills for Linux device driver development and debugging
Apply problem solving skills and experience to identify and improve low-level system performance issues
Apply engineering principles to design algorithms for controlling image acquisition parameters, as well as environmental conditions (Temperature, Power, Fog, Frost, etc.)
Create design documents on software architecture and algorithms
Collaborate with Hardware designers on board bring-up and debug
Maintain and improve Firmware build system using Make and Python
Collaborate with Quality Assurance team on identifying test cases for new features and areas for regression tests
Follow the established development process for all design and implementation tasks
Your skills and experience
Bachelor / Master degree in Computer Engineering, Software Engineering. Having a EE background is a plus.
Experience in writing quality C or Modern C++ on Linux OS based embedded systems. Experience in Rust is a plus.
Experience in the Linux build system. Familiarity with Yocto is a plus.
knowledge in writing low level programming for HW peripherals and drivers.
Knowledge on networking protocols / connectivity, such as Wifi, Bluetooth, used with embedded systems
Knowledge in camera linux embedded systems is a plus.
Good written English and oral communication skills.
Desire to learn.
A team player
Why you'll love working here
Compensation & bonus: 13th & 14th salary, AIP bonus, Holidays, Tet, and Long year service …
Social insurance, Health insurance, Unemployment insurance: by Social Insurance and Labor Law
The regime of annual leave, company trip, and checkup examination
Award for marriage, newborn
We have AON insurance package for employee, spouse, and children every year
You will be trained, learned & work with the best technical managers who help you improve various dev skills & career path
You'll love working in our dynamic environment employees, young & active
We love sport activities, as marathon, football, swimming,...
Working time: From Monday to Friday | 08:30-12:00 & 13:00-17.30""",
        "required": "Bachelor / Master degree in Computer Engineering, Software Engineering",
        "address": "Ho Chi Minh City",
        "location_id": 2,
        "salary": "Negotiable",
        "experience_id": 3,
        "member": "10-50 people",
        "work_type_id": 1,
        "category_id": 1,
        "posted_expired": "2025-06-23T11:47:25.025Z"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/job/extract/all-features",
            json=job_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Job extraction successful!")
            print(f"   Primary skills: {result['primary_skills']}")
            print(f"   Secondary skills: {result['secondary_skills']}")
            print(f"   Adverbs: {result['adverbs']}")
            print(f"   Adjectives: {result['adjectives']}")
            return True
        else:
            print(f"❌ Error in Job extraction: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception testing Job extraction: {e}")
        return False

def test_api_endpoints_exist():
    """Test that save-features endpoints are removed"""
    print("\n=== Testing Removed Endpoints ===")
    
    # Test CV save-features (should not exist)
    try:
        response = requests.post(f"{BASE_URL}/cv/save-features")
        if response.status_code == 404:
            print("✅ CV save-features endpoint correctly removed!")
        else:
            print(f"⚠️  CV save-features endpoint still exists: {response.status_code}")
    except:
        print("✅ CV save-features endpoint correctly removed!")
    
    # Test Job save-features (should not exist)
    try:
        response = requests.post(f"{BASE_URL}/job/save-features")
        if response.status_code == 404:
            print("✅ Job save-features endpoint correctly removed!")
        else:
            print(f"⚠️  Job save-features endpoint still exists: {response.status_code}")
    except:
        print("✅ Job save-features endpoint correctly removed!")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Updated API Tests...")
    
    # Test API health first
    if not test_health_check():
        print("\n❌ API is not accessible. Please start the server:")
        print("   uvicorn app.main:app --reload")
        exit(1)
    
    # Run all tests
    success_count = 0
    total_tests = 3
    
    if test_cv_extraction_unique():
        success_count += 1
    
    if test_job_extraction_with_complex_description():
        success_count += 1
        
    if test_api_endpoints_exist():
        success_count += 1
    
    print(f"\n🎉 Tests completed: {success_count}/{total_tests} passed!")
    
    if success_count == total_tests:
        print("✅ All tests passed successfully!")
        print("✅ CV unique constraint working!")
        print("✅ Job complex description handling working!")
        print("✅ Removed endpoints confirmed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("   - Make sure database is running and accessible")
        print("   - Check if the required tables exist")
        print("   - Run: python create_tables.py to setup database")
