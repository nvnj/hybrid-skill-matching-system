#!/usr/bin/env python3
"""
Test script for the web application skill matching functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_api_endpoints():
    """Test all API endpoints"""
    print("🧪 Testing Skills-Based Job Matcher API")
    print("=" * 50)
    
    # Test system status
    print("\n1. Testing System Status...")
    try:
        response = requests.get(f"{BASE_URL}/api/system-status")
        if response.status_code == 200:
            status = response.json()
            print("✅ System Status OK")
            print(f"   - Courses: {status['status']['total_courses']}")
            print(f"   - Jobs: {status['status']['total_jobs']}")
            print(f"   - Skills: {status['status']['total_unique_skills']}")
            print(f"   - Trained: {status['status']['is_trained']}")
        else:
            print("❌ System Status Failed")
            return False
    except Exception as e:
        print(f"❌ System Status Error: {e}")
        return False
    
    # Test courses endpoint
    print("\n2. Testing Courses Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/courses")
        if response.status_code == 200:
            courses_data = response.json()
            courses = courses_data['courses']
            print(f"✅ Courses OK - Found {len(courses)} courses")
            
            # Show first few courses
            for i, course in enumerate(courses[:3]):
                print(f"   - {course['course_code']}: {course['title']} ({course['skill_count']} skills)")
            
            if len(courses) > 3:
                print(f"   ... and {len(courses) - 3} more courses")
                
        else:
            print("❌ Courses Failed")
            return False
    except Exception as e:
        print(f"❌ Courses Error: {e}")
        return False
    
    # Test job matching with sample courses
    print("\n3. Testing Job Matching...")
    try:
        # Use first 2 courses for testing
        test_courses = [course['course_code'] for course in courses[:2]]
        print(f"   Testing with courses: {test_courses}")
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/match-jobs", 
                               json={"courses": test_courses, "top_k": 3})
        end_time = time.time()
        
        if response.status_code == 200:
            matches_data = response.json()
            matches = matches_data['matches']
            print(f"✅ Job Matching OK - Found {len(matches)} matches")
            print(f"   Response time: {(end_time - start_time):.2f} seconds")
            
            # Show match details
            for i, match in enumerate(matches):
                print(f"\n   🎯 Match #{i+1}:")
                print(f"      Job: {match['title']}")
                print(f"      Industry: {match['industry']}")
                print(f"      Match Score: {(match['score'] * 100):.1f}%")
                print(f"      Skills Matched: {match['matched_count']}/{match['total_job_skills']}")
                
                # Show top matched skills
                if match['matched_skills']:
                    print("      Top Skills:")
                    for skill_match in match['matched_skills'][:3]:
                        print(f"        - {skill_match['job_skill']} ({(skill_match['score'] * 100):.0f}%)")
        else:
            print(f"❌ Job Matching Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Job Matching Error: {e}")
        return False
    
    # Test course skills endpoint
    print("\n4. Testing Course Skills Endpoint...")
    try:
        test_course = courses[0]['course_code']
        response = requests.get(f"{BASE_URL}/api/course-skills/{test_course}")
        if response.status_code == 200:
            skills_data = response.json()
            skills = skills_data['skills']
            print(f"✅ Course Skills OK - {len(skills)} skills for {test_course}")
            
            # Show some skills
            for skill in skills[:3]:
                print(f"   - {skill['term']} ({skill['category']})")
                
        else:
            print("❌ Course Skills Failed")
            return False
    except Exception as e:
        print(f"❌ Course Skills Error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All API tests passed successfully!")
    print("\n📊 System Performance Summary:")
    print(f"   - Total response time: {(end_time - start_time):.2f} seconds")
    print(f"   - Courses loaded: {len(courses)}")
    print(f"   - Job matches found: {len(matches)}")
    print(f"   - Average match score: {np.mean([m['score'] for m in matches]) * 100:.1f}%")
    
    return True

def test_matching_quality():
    """Test the quality of skill matching"""
    print("\n🔍 Testing Matching Quality")
    print("=" * 50)
    
    try:
        # Get courses
        response = requests.get(f"{BASE_URL}/api/courses")
        courses = response.json()['courses']
        
        # Test different course combinations
        test_scenarios = [
            {"name": "Single Course", "courses": [courses[0]['course_code']]},
            {"name": "Two Courses", "courses": [courses[0]['course_code'], courses[1]['course_code']]},
            {"name": "Three Courses", "courses": [c['course_code'] for c in courses[:3]]},
        ]
        
        for scenario in test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            print(f"   Courses: {scenario['courses']}")
            
            response = requests.post(f"{BASE_URL}/api/match-jobs", 
                                   json={"courses": scenario['courses'], "top_k": 5})
            
            if response.status_code == 200:
                matches = response.json()['matches']
                
                if matches:
                    scores = [m['score'] for m in matches]
                    match_counts = [m['matched_count'] for m in matches]
                    
                    print(f"   ✅ Found {len(matches)} job matches")
                    print(f"   📈 Score range: {min(scores)*100:.1f}% - {max(scores)*100:.1f}%")
                    print(f"   🎯 Avg skills matched: {np.mean(match_counts):.1f}")
                    print(f"   🏆 Best match: {matches[0]['title']} ({matches[0]['score']*100:.1f}%)")
                else:
                    print("   ⚠️ No matches found")
            else:
                print(f"   ❌ Error: {response.status_code}")
        
        print(f"\n✅ Matching quality test completed")
        
    except Exception as e:
        print(f"❌ Matching quality test error: {e}")

if __name__ == "__main__":
    import numpy as np
    
    print("🚀 Starting Web Application Tests")
    print(f"🌐 Target URL: {BASE_URL}")
    
    # Wait a moment for server to be ready
    print("\n⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Run tests
    success = test_api_endpoints()
    
    if success:
        test_matching_quality()
        print("\n🎊 All tests completed successfully!")
        print("\n💡 You can now:")
        print(f"   1. Open {BASE_URL} in your browser")
        print("   2. Select courses and find matching jobs")
        print("   3. Explore the skill matching results")
    else:
        print("\n❌ Some tests failed. Check the server logs.")
