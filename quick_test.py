#!/usr/bin/env python3
"""
Quick test script for skill extraction - processes only first few entries
"""

import os
import pandas as pd
from extraction import BatchSkillProcessor, LLMProvider

def quick_test_extraction():
    """Test the extraction with only the first 2 entries"""
    
    print("QUICK TEST: Skill Extraction System")
    print("=" * 50)
    print("Processing only first 2 entries from each file\n")
    
    # Test with all three models
    providers = [LLMProvider.GPT4, LLMProvider.DEEPSEEK, LLMProvider.LLAMA]
    
    for provider in providers:
        print(f"\n{'='*20} Testing {provider.value} {'='*20}")
        
        try:
            # Initialize processor
            processor = BatchSkillProcessor(
                provider=provider,
                rate_limit_delay=0.5  # Shorter delay for testing
            )
            
            # Test with jobs CSV (only first 2 entries)
            if os.path.exists('Jobs.csv'):
                print(f"\nProcessing Jobs.csv with {provider.value} (first 2 entries)...")
                job_results = processor.process_jobs_csv(
                    'Jobs.csv',
                    output_file=f'quick_test_job_skills_{provider.value}.csv',
                    sample_size=2
                )
                
                # Show results summary
                if len(job_results) > 0:
                    total_entries = len(job_results)
                    skills_count = len(job_results[job_results['is_skill'] == True])
                    print(f"Results: {total_entries} total entries, {skills_count} skills identified")
                    
                    # Show sample skills
                    skills_only = job_results[job_results['is_skill'] == True]
                    print(f"Sample skills extracted:")
                    for _, row in skills_only.head(5).iterrows():
                        category = row['skill_category'] if pd.notna(row['skill_category']) else 'NO_CATEGORY'
                        print(f"  • {row['term']} ({category})")
                    
                    # Check for duplicates
                    duplicates = skills_only['term'].value_counts()
                    duplicates = duplicates[duplicates > 1]
                    if len(duplicates) > 0:
                        print(f"⚠️  Duplicates found: {list(duplicates.index)}")
                    else:
                        print("✅ No duplicates found")
                    
                    # Check for missing categories
                    missing_categories = skills_only['skill_category'].isna().sum()
                    if missing_categories > 0:
                        print(f"⚠️  Missing categories: {missing_categories}")
                    else:
                        print("✅ All skills have categories")
                else:
                    print("No results returned")
            
            # Test with courses CSV (only first 2 entries)
            if os.path.exists('Learning_Outcomes_with_Skills.csv'):
                print(f"\nProcessing Learning_Outcomes_with_Skills.csv with {provider.value} (first 2 entries)...")
                course_results = processor.process_courses_csv(
                    'Learning_Outcomes_with_Skills.csv',
                    output_file=f'quick_test_course_skills_{provider.value}.csv',
                    sample_size=2
                )
                
                # Show results summary
                if len(course_results) > 0:
                    total_entries = len(course_results)
                    skills_count = len(course_results[course_results['is_skill'] == True])
                    print(f"Results: {total_entries} total entries, {skills_count} skills identified")
                    
                    # Show sample skills
                    skills_only = course_results[course_results['is_skill'] == True]
                    print(f"Sample skills extracted:")
                    for _, row in skills_only.head(5).iterrows():
                        category = row['skill_category'] if pd.notna(row['skill_category']) else 'NO_CATEGORY'
                        print(f"  • {row['term']} ({category})")
                    
                    # Check for issues
                    duplicates = skills_only['term'].value_counts()
                    duplicates = duplicates[duplicates > 1]
                    if len(duplicates) > 0:
                        print(f"⚠️  Duplicates found: {list(duplicates.index)}")
                    else:
                        print("✅ No duplicates found")
                    
                    missing_categories = skills_only['skill_category'].isna().sum()
                    if missing_categories > 0:
                        print(f"⚠️  Missing categories: {missing_categories}")
                    else:
                        print("✅ All skills have categories")
                else:
                    print("No results returned")
                
        except Exception as e:
            print(f"❌ Error testing {provider.value}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 50)
    print("Quick test completed!")
    print("Check the quick_test_*_skills_*.csv files for detailed results.")

def compare_quick_results():
    """Compare the quick test results"""
    
    print("\n" + "=" * 50)
    print("COMPARING QUICK TEST RESULTS")
    print("=" * 50)
    
    providers = ['gpt-4', 'deepseek', 'llama']
    
    for task in ['job', 'course']:
        print(f"\n{task.upper()} SKILLS COMPARISON:")
        print("-" * 30)
        
        for provider in providers:
            filename = f'quick_test_{task}_skills_{provider}.csv'
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                skills = df[df['is_skill'] == True]
                
                print(f"\n{provider.upper()}:")
                print(f"  Total skills: {len(skills)}")
                
                # Category breakdown
                if len(skills) > 0:
                    categories = skills['skill_category'].value_counts()
                    print(f"  Categories: {dict(categories)}")
                    
                    # Check for specific issues
                    missing_cat = skills['skill_category'].isna().sum()
                    if missing_cat > 0:
                        print(f"  ⚠️  Missing categories: {missing_cat}")
                    
                    # Show unique skills
                    unique_skills = skills['term'].unique()
                    print(f"  Unique skills: {list(unique_skills)}")
            else:
                print(f"\n{provider.upper()}: File not found")

if __name__ == "__main__":
    # Run quick tests
    quick_test_extraction()
    compare_quick_results()
