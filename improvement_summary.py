#!/usr/bin/env python3
"""
Summary of improvements made to the skill extraction system
"""

import pandas as pd
import os

def summarize_improvements():
    """Compare original vs improved outputs"""
    
    print("SKILL EXTRACTION SYSTEM - IMPROVEMENTS SUMMARY")
    print("=" * 60)
    
    # Compare original vs improved outputs
    comparisons = [
        {
            'task': 'Job Skills',
            'original_files': ['job_skills_extracted_deepseek.csv', 'job_skills_extracted_llama.csv', 'job_skills_extracted_gpt-4.csv'],
            'improved_files': ['quick_test_job_skills_deepseek.csv', 'quick_test_job_skills_llama.csv', 'quick_test_job_skills_gpt-4.csv']
        },
        {
            'task': 'Course Skills', 
            'original_files': ['course_skills_extracted_deepseek.csv', 'course_skills_extracted_llama.csv', 'course_skills_extracted_gpt-4.csv'],
            'improved_files': ['quick_test_course_skills_deepseek.csv', 'quick_test_course_skills_llama.csv', 'quick_test_course_skills_gpt-4.csv']
        }
    ]
    
    for comparison in comparisons:
        print(f"\n{comparison['task'].upper()} ANALYSIS:")
        print("-" * 40)
        
        for i, (original_file, improved_file) in enumerate(zip(comparison['original_files'], comparison['improved_files'])):
            model_name = ['DeepSeek', 'LLaMA', 'GPT-4'][i]
            print(f"\n{model_name} Results:")
            
            # Analyze original file (if exists)
            if os.path.exists(original_file):
                try:
                    orig_df = pd.read_csv(original_file)
                    orig_issues = analyze_issues(orig_df, "Original")
                    print(f"  Original: {len(orig_df)} entries, Issues: {orig_issues}")
                except Exception as e:
                    print(f"  Original: Error reading file - {e}")
            else:
                print(f"  Original: File not found")
            
            # Analyze improved file
            if os.path.exists(improved_file):
                try:
                    improved_df = pd.read_csv(improved_file)
                    improved_issues = analyze_issues(improved_df, "Improved")
                    print(f"  Improved: {len(improved_df)} entries, Issues: {improved_issues}")
                    
                    # Show sample skills
                    skills = improved_df[improved_df['is_skill'] == True]
                    if len(skills) > 0:
                        print(f"  Sample skills: {', '.join(skills['term'].head(3).tolist())}")
                        
                        # Category distribution
                        categories = skills['skill_category'].value_counts().head(3)
                        print(f"  Top categories: {dict(categories)}")
                    
                except Exception as e:
                    print(f"  Improved: Error reading file - {e}")
            else:
                print(f"  Improved: File not found")

def analyze_issues(df, label):
    """Analyze issues in a dataframe"""
    issues = []
    
    # Check for duplicates
    duplicates = df['term'].value_counts()
    duplicates = duplicates[duplicates > 1]
    if len(duplicates) > 0:
        issues.append(f"{len(duplicates)} duplicates")
    
    # Check for formatting issues
    formatting_issues = 0
    for term in df['term']:
        if pd.notna(term) and ('**' in str(term) or '***' in str(term)):
            formatting_issues += 1
    
    if formatting_issues > 0:
        issues.append(f"{formatting_issues} formatting issues")
    
    # Check for missing categories
    skills_df = df[df['is_skill'] == True]
    if len(skills_df) > 0:
        missing_categories = skills_df['skill_category'].isna().sum()
        if missing_categories > 0:
            issues.append(f"{missing_categories} missing categories")
    
    return "; ".join(issues) if issues else "None"

def show_key_improvements():
    """Show the key improvements made"""
    
    print("\n" + "=" * 60)
    print("KEY IMPROVEMENTS IMPLEMENTED")
    print("=" * 60)
    
    improvements = [
        "✅ DUPLICATE REMOVAL:",
        "   - Added normalized term comparison (case-insensitive)",
        "   - Tracks seen terms to prevent duplicates",
        "   - Removes terms with ** formatting that duplicate clean terms",
        "",
        "✅ FORMATTING CLEANUP:",
        "   - Removes ** and other markdown formatting from terms",
        "   - Cleans numbering and prefixes",
        "   - Normalizes whitespace",
        "",
        "✅ CATEGORY EXTRACTION FIX:",
        "   - Improved regex to extract categories from reasoning",
        "   - Added fallback category inference from reasoning text",
        "   - Ensures all skills have categories assigned",
        "",
        "✅ MODEL-SPECIFIC PROMPTS:",
        "   - DeepSeek: Explicit instructions to avoid ** formatting",
        "   - LLaMA: Emphasis on including categories in parentheses", 
        "   - GPT-4: Maintained high-quality consistent format",
        "",
        "✅ VALIDATION & CLEANUP:",
        "   - Post-processing validation step",
        "   - Additional duplicate checking",
        "   - Quality control on extracted terms",
        "",
        "✅ NEW SKILL CATEGORIES:",
        "   - Added 'operational_skill' for maintenance/safety tasks",
        "   - Added 'trade_skill' for specialized craft skills",
        "   - Models can create new categories when needed",
        "",
        "✅ BETTER ERROR HANDLING:",
        "   - Improved logging and debugging",
        "   - Graceful handling of extraction failures",
        "   - Better temperature and parameter tuning"
    ]
    
    for improvement in improvements:
        print(improvement)

def next_steps():
    """Show recommended next steps"""
    
    print("\n" + "=" * 60)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 60)
    
    steps = [
        "1. 🚀 FULL DATASET PROCESSING:",
        "   python3 extraction.py --jobs Jobs.csv --all-models",
        "   python3 extraction.py --courses Learning_Outcomes_with_Skills.csv --all-models",
        "",
        "2. 📊 SKILL MATCHING ANALYSIS:",
        "   - Implement skill matching between job and course skills",
        "   - Create similarity scoring for skill alignment",
        "   - Generate skill gap analysis reports",
        "",
        "3. 🔍 QUALITY ASSESSMENT:",
        "   - Manual review of extracted skills for accuracy",
        "   - Inter-model agreement analysis",
        "   - Refinement of category definitions",
        "",
        "4. 📈 ADVANCED FEATURES:",
        "   - Skill level classification (beginner/intermediate/advanced)",
        "   - Skill importance scoring",
        "   - Industry-specific skill categorization",
        "",
        "5. 🎯 DEPLOYMENT:",
        "   - Create automated pipeline for new job/course data",
        "   - Build skill recommendation system",
        "   - Develop matching dashboard"
    ]
    
    for step in steps:
        print(step)

if __name__ == "__main__":
    summarize_improvements()
    show_key_improvements()
    next_steps()
