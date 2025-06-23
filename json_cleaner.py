import json
import re

def clean_json_string(text):
    """Clean text to make it JSON-safe"""
    # Replace various line endings with \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Escape special characters for JSON
    text = text.replace('\\', '\\\\')  # Escape backslashes
    text = text.replace('"', '\\"')   # Escape quotes
    text = text.replace('\n', '\\n')  # Escape newlines
    text = text.replace('\t', '\\t')  # Escape tabs
    
    return text

# Your original description
original_description = """    Job description
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
    A team player Why you will love working here
    Compensation & bonus: 13th & 14th salary, AIP bonus, Holidays, Tet, and Long year service …
    Social insurance, Health insurance, Unemployment insurance: by Social Insurance and Labor Law
    The regime of annual leave, company trip, and checkup examination
    Award for marriage, newborn
    We have AON insurance package for employee, spouse, and children every year
    You will be trained, learned & work with the best technical managers who help you improve various dev skills & career path
    You'll love working in our dynamic environment employees, young & active
    We love sport activities, as marathon, football, swimming,...
    Working time: From Monday to Friday | 08:30-12:00 & 13:00-17.30"""

# Clean the description
cleaned_description = clean_json_string(original_description)

# Create the JSON object
job_data = {
    "employer_id": 1,
    "title": "abc intern",
    "description": cleaned_description,
    "required": "string",
    "address": "string",
    "location_id": 1,
    "salary": "20000",
    "experience_id": 1,
    "member": "string",
    "work_type_id": 1,
    "category_id": 1,
    "posted_expired": "string"
}

# Convert to JSON and validate
try:
    json_string = json.dumps(job_data, indent=2, ensure_ascii=False)
    print("✅ Valid JSON created!")
    print("\n" + "="*50)
    print("CLEANED JSON:")
    print("="*50)
    print(json_string)
    
    # Test parsing back
    parsed = json.loads(json_string)
    print("\n✅ JSON validation successful!")
    
    # Save to file
    with open('cleaned_job_request.json', 'w', encoding='utf-8') as f:
        f.write(json_string)
    print("✅ Saved to 'cleaned_job_request.json'")
    
except json.JSONEncodeError as e:
    print(f"❌ JSON Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
