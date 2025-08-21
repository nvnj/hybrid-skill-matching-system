#!/usr/bin/env python3
"""
Flask backend for skill matching web application
Implements hybrid skill matching approach as recommended in state-of-the-art research
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import json
from typing import List, Dict, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class HybridSkillMatcher:
    """
    Hybrid skill matching system combining semantic embeddings with structured knowledge
    Based on state-of-the-art research recommendations for small-to-medium scale applications
    """
    
    def __init__(self):
        self.course_skills = {}
        self.job_skills = {}
        self.skill_embeddings = None
        self.tfidf_vectorizer = None
        self.skill_to_category = {}
        self.course_data = None
        self.job_data = None
        self.is_trained = False
        
    def load_data(self, course_skills_file: str, job_skills_file: str):
        """Load extracted skills data from CSV files"""
        try:
            # Load course skills
            course_df = pd.read_csv(course_skills_file)
            course_skills_only = course_df[course_df['is_skill'] == True]
            
            # Group skills by course
            for _, row in course_skills_only.iterrows():
                course_code = row['course_code']
                if course_code not in self.course_skills:
                    self.course_skills[course_code] = []
                
                skill_info = {
                    'term': row['term'],
                    'category': row['skill_category'],
                    'reasoning': row['reasoning']
                }
                self.course_skills[course_code].append(skill_info)
                self.skill_to_category[row['term']] = row['skill_category']
            
            # Load job skills
            job_df = pd.read_csv(job_skills_file)
            job_skills_only = job_df[job_df['is_skill'] == True]
            
            # Group skills by job
            for _, row in job_skills_only.iterrows():
                job_id = row['job_id']
                if job_id not in self.job_skills:
                    self.job_skills[job_id] = []
                
                skill_info = {
                    'term': row['term'],
                    'category': row['skill_category'],
                    'reasoning': row['reasoning']
                }
                self.job_skills[job_id].append(skill_info)
                self.skill_to_category[row['term']] = row['skill_category']
            
            logger.info(f"Loaded {len(self.course_skills)} courses and {len(self.job_skills)} jobs")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def load_job_metadata(self, jobs_file: str):
        """Load job metadata for displaying job information"""
        try:
            self.job_data = pd.read_csv(jobs_file)
            logger.info(f"Loaded metadata for {len(self.job_data)} jobs")
            return True
        except Exception as e:
            logger.error(f"Error loading job metadata: {e}")
            return False
    
    def load_course_metadata(self, courses_file: str):
        """Load course metadata for displaying course information"""
        try:
            self.course_data = pd.read_csv(courses_file, encoding='cp1252')
            logger.info(f"Loaded metadata for {len(self.course_data)} courses")
            return True
        except Exception as e:
            logger.error(f"Error loading course metadata: {e}")
            return False
    
    def train_embeddings(self):
        """
        Train domain-specific embeddings using TF-IDF
        Following state-of-the-art recommendation for small-scale implementations
        """
        try:
            # Collect all skills
            all_skills = set()
            for skills in self.course_skills.values():
                for skill in skills:
                    all_skills.add(skill['term'])
            
            for skills in self.job_skills.values():
                for skill in skills:
                    all_skills.add(skill['term'])
            
            all_skills = list(all_skills)
            
            # Create TF-IDF embeddings
            # Using character n-grams to capture semantic similarity as recommended
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 3),
                analyzer='char_wb',
                max_features=5000
            )
            
            self.skill_embeddings = self.tfidf_vectorizer.fit_transform(all_skills)
            self.skill_list = all_skills
            
            self.is_trained = True
            logger.info(f"Trained embeddings for {len(all_skills)} unique skills")
            return True
            
        except Exception as e:
            logger.error(f"Error training embeddings: {e}")
            return False
    
    def calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate semantic similarity between two skills"""
        if not self.is_trained:
            return 0.0
        
        try:
            if skill1 not in self.skill_list or skill2 not in self.skill_list:
                return 0.0
            
            idx1 = self.skill_list.index(skill1)
            idx2 = self.skill_list.index(skill2)
            
            similarity = cosine_similarity(
                self.skill_embeddings[idx1:idx1+1],
                self.skill_embeddings[idx2:idx2+1]
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_category_bonus(self, cat1: str, cat2: str) -> float:
        """
        Apply ontological enhancement - bonus for same skill category
        Following hybrid approach recommendations
        """
        if cat1 == cat2:
            return 0.2  # 20% bonus for same category
        return 0.0
    
    def match_courses_to_jobs(self, selected_courses: List[str], top_k: int = 5) -> List[Dict]:
        """
        Match selected courses to jobs using hybrid approach
        Combines semantic similarity with ontological structure
        """
        if not self.is_trained:
            return []
        
        try:
            # Collect all skills from selected courses
            student_skills = []
            for course_code in selected_courses:
                if course_code in self.course_skills:
                    student_skills.extend(self.course_skills[course_code])
            
            if not student_skills:
                return []
            
            # Calculate job scores
            job_scores = []
            
            for job_id, job_skills in self.job_skills.items():
                total_score = 0.0
                matched_skills = []
                
                for job_skill in job_skills:
                    best_match_score = 0.0
                    best_match_student_skill = None
                    
                    for student_skill in student_skills:
                        # Semantic similarity
                        semantic_score = self.calculate_skill_similarity(
                            student_skill['term'], job_skill['term']
                        )
                        
                        # Ontological enhancement
                        category_bonus = self.get_category_bonus(
                            student_skill['category'], job_skill['category']
                        )
                        
                        # Combined score
                        combined_score = semantic_score + category_bonus
                        
                        if combined_score > best_match_score:
                            best_match_score = combined_score
                            best_match_student_skill = student_skill
                    
                    # Threshold for considering a skill matched (as per research)
                    if best_match_score > 0.3:  # 30% threshold
                        total_score += best_match_score
                        matched_skills.append({
                            'job_skill': job_skill['term'],
                            'student_skill': best_match_student_skill['term'],
                            'score': best_match_score,
                            'job_category': job_skill['category'],
                            'student_category': best_match_student_skill['category']
                        })
                
                # Normalize by number of job skills (prevents bias toward jobs with many skills)
                if len(job_skills) > 0:
                    normalized_score = total_score / len(job_skills)
                    match_percentage = (len(matched_skills) / len(job_skills)) * 100
                    
                    job_scores.append({
                        'job_id': job_id,
                        'score': normalized_score,
                        'match_percentage': match_percentage,
                        'matched_skills': matched_skills,
                        'total_job_skills': len(job_skills),
                        'matched_count': len(matched_skills)
                    })
            
            # Sort by score and return top-k
            job_scores.sort(key=lambda x: x['score'], reverse=True)
            return job_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error matching courses to jobs: {e}")
            return []
    
    def get_job_details(self, job_id: int) -> Dict:
        """Get job details from metadata"""
        if self.job_data is None:
            return {}
        
        try:
            job_row = self.job_data[self.job_data['Job_Id'] == job_id]
            if len(job_row) == 0:
                return {}
            
            job = job_row.iloc[0]
            return {
                'job_id': int(job['Job_Id']),
                'title': job.get('inferred_job_title', 'Unknown'),
                'industry': job.get('inferred_co_industry', 'Unknown'),
                'description': job.get('job_description', '')[:500] + '...' if len(job.get('job_description', '')) > 500 else job.get('job_description', ''),
                'skills_overview': job.get('JobsPiks_skills', '')
            }
        except Exception as e:
            logger.error(f"Error getting job details: {e}")
            return {}
    
    def get_course_details(self, course_code: str) -> Dict:
        """Get course details from metadata"""
        if self.course_data is None:
            return {'course_code': course_code, 'title': course_code}
        
        try:
            course_row = self.course_data[self.course_data['Course_Code'] == course_code]
            if len(course_row) == 0:
                return {'course_code': course_code, 'title': course_code}
            
            course = course_row.iloc[0]
            return {
                'course_code': course_code,
                'title': course.get('Course', course_code),
                'degree': course.get('Degree', 'Unknown'),
                'description': course.get('Combined_Text', '')[:200] + '...' if len(course.get('Combined_Text', '')) > 200 else course.get('Combined_Text', '')
            }
        except Exception as e:
            logger.error(f"Error getting course details: {e}")
            return {'course_code': course_code, 'title': course_code}

# Initialize the matcher
matcher = HybridSkillMatcher()

@app.route('/')
def index():
    """Serve the main application page"""
    return app.send_static_file('index.html')

@app.route('/api/courses', methods=['GET'])
def get_courses():
    """Get list of available courses"""
    try:
        courses = []
        for course_code in matcher.course_skills.keys():
            course_details = matcher.get_course_details(course_code)
            course_details['skill_count'] = len(matcher.course_skills[course_code])
            courses.append(course_details)
        
        return jsonify({
            'success': True,
            'courses': courses
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/match-jobs', methods=['POST'])
def match_jobs():
    """Match selected courses to jobs"""
    try:
        data = request.get_json()
        selected_courses = data.get('courses', [])
        top_k = data.get('top_k', 5)
        
        if not selected_courses:
            return jsonify({
                'success': False,
                'error': 'No courses selected'
            }), 400
        
        # Get job matches
        job_matches = matcher.match_courses_to_jobs(selected_courses, top_k)
        
        # Enrich with job details
        enriched_matches = []
        for match in job_matches:
            job_details = matcher.get_job_details(match['job_id'])
            match.update(job_details)
            enriched_matches.append(match)
        
        return jsonify({
            'success': True,
            'matches': enriched_matches,
            'selected_courses': selected_courses,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in match_jobs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/course-skills/<course_code>', methods=['GET'])
def get_course_skills(course_code):
    """Get skills for a specific course"""
    try:
        if course_code not in matcher.course_skills:
            return jsonify({
                'success': False,
                'error': 'Course not found'
            }), 404
        
        skills = matcher.course_skills[course_code]
        course_details = matcher.get_course_details(course_code)
        
        return jsonify({
            'success': True,
            'course': course_details,
            'skills': skills
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system-status', methods=['GET'])
def system_status():
    """Get system status and statistics"""
    try:
        return jsonify({
            'success': True,
            'status': {
                'is_trained': matcher.is_trained,
                'total_courses': len(matcher.course_skills),
                'total_jobs': len(matcher.job_skills),
                'total_unique_skills': len(matcher.skill_list) if matcher.is_trained else 0,
                'loaded_job_metadata': matcher.job_data is not None,
                'loaded_course_metadata': matcher.course_data is not None
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def initialize_system():
    """Initialize the skill matching system"""
    logger.info("Initializing Hybrid Skill Matching System...")
    
    # Load data files
    course_skills_file = 'course_skills_extracted_gpt-4.csv'  # Use best performing model
    job_skills_file = 'job_skills_extracted_gpt-4.csv'
    jobs_metadata_file = 'Jobs.csv'
    courses_metadata_file = 'Learning_Outcomes_with_Skills.csv'
    
    # Check if files exist
    for file in [course_skills_file, job_skills_file, jobs_metadata_file, courses_metadata_file]:
        if not os.path.exists(file):
            logger.warning(f"File not found: {file}")
    
    # Load data
    success = True
    success &= matcher.load_data(course_skills_file, job_skills_file)
    success &= matcher.load_job_metadata(jobs_metadata_file)
    success &= matcher.load_course_metadata(courses_metadata_file)
    
    if success:
        # Train embeddings
        success &= matcher.train_embeddings()
    
    if success:
        logger.info("System initialized successfully!")
    else:
        logger.error("System initialization failed!")
    
    return success

if __name__ == '__main__':
    # Initialize the system
    if initialize_system():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize system. Exiting.")
        exit(1)
