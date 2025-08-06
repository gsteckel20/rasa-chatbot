import json
import os
import re

INPUT_PATH = os.path.join('scrapy', 'output', 'uga_courses.json')
OUTPUT_PATH = os.path.join('cleaned_data', 'uga_courses_update.sql')

# Updated regex to capture the full department including parentheses and the full course number string
COURSE_ID_REGEX = re.compile(r'^(?P<dept>[A-Z]{2,4}(?:\([^)]+\))*)\s*(?P<num>[^\s]+)')

# Updated regex to extract credit hours as a string (handles more cases)
CREDIT_HOURS_REGEX = re.compile(r'(\d+(?:-\d+)?(?:\.\d+)?)')

def extract_first_number(num_str):
    # Split on / or - and take the first part
    first = re.split(r'[/-]', num_str)[0]
    # Remove any trailing non-digit characters (e.g., S, E, H, L)
    digits = re.match(r'(\d{4})', first)
    return digits.group(1) if digits else first

def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as infile:
        courses = json.load(infile)
    
    sql_lines = []
    skipped_courses = []
    
    for course in courses:
        course_id = course.get('course_id', '')
        credit_hours = course.get('credit_hours', '')
        
        # Try to match the course ID
        match = COURSE_ID_REGEX.match(course_id)
        if not match:
            skipped_courses.append(course_id)
            continue
            
        dept = match.group('dept')
        num_raw = match.group('num')
        num = extract_first_number(num_raw)
        
        # Extract the credit hours as a string
        ch_match = CREDIT_HOURS_REGEX.search(credit_hours)
        if not ch_match:
            skipped_courses.append(f"{course_id} (no credit hours)")
            continue
            
        ch_str = ch_match.group(1)
        
        sql = (
            f'UPDATE UGACourseRate.Class\n'
            f'SET credithours = \'{ch_str}\'\n'
            f'WHERE classnum = {num} and department = \'{dept}\';'
        )
        sql_lines.append(sql)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(sql_lines))
    
    print(f"Wrote {len(sql_lines)} SQL update statements to {OUTPUT_PATH}")
    print(f"Skipped {len(skipped_courses)} courses that couldn't be parsed")
    
    if skipped_courses:
        print("\nFirst 10 skipped courses:")
        for course in skipped_courses[:10]:
            print(f"  - {course}")

if __name__ == '__main__':
    main() 