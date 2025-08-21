#!/usr/bin/env python3
"""
Test script for the improved skill extraction system
"""

import os
import pandas as pd
from extraction import BatchSkillProcessor, LLMProvider

def test_extraction_sample():
    """Test the extraction with a small sample"""
    
    # Test with a small sample from each model
    providers = [LLMProvider.GPT4, LLMProvider.DEEPSEEK, LLMProvider.LLAMA]
    
    print("Testing improved skill extraction system")
    print("=" * 60)
    
    for provider in providers:
        print(f"\nTesting {provider.value}...")
        
        try:
            # Initialize processor
            processor = BatchSkillProcessor(
                provider=provider,
                rate_limit_delay=0.5  # Shorter delay for testing
            )
            
            # Test with jobs CSV (sample size 1)
            if os.path.exists('Jobs.csv'):
                print(f"Processing Jobs.csv with {provider.value}...")
                job_results = processor.process_jobs_csv(
                    'Jobs.csv',
                    output_file=f'test_job_skills_{provider.value}.csv',
                    sample_size=1
                )
                
                # Show sample results
                if len(job_results) > 0:
                    print(f"Sample results from {provider.value}:")
                    skills_only = job_results[job_results['is_skill'] == True].head(5)
                    for _, row in skills_only.iterrows():
                        print(f"  {row['term']} | {row['skill_category']} | {row['reasoning']}")
                
            # Test with courses CSV (sample size 1)
            if os.path.exists('Learning_Outcomes_with_Skills.csv'):
                print(f"Processing Learning_Outcomes_with_Skills.csv with {provider.value}...")
                course_results = processor.process_courses_csv(
                    'Learning_Outcomes_with_Skills.csv',
                    output_file=f'test_course_skills_{provider.value}.csv',
                    sample_size=1
                )
                
        except Exception as e:
            print(f"Error testing {provider.value}: {e}")
            continue
    
    print("\nTesting completed. Check the test_*_skills_*.csv files for results.")

def compare_outputs():
    """Compare outputs from different models"""
    
    print("\nComparing model outputs...")
    
    providers = ['gpt-4', 'deepseek', 'llama']
    
    for task in ['job', 'course']:
        print(f"\n{task.upper()} SKILLS COMPARISON:")
        print("-" * 40)
        
        all_skills = {}
        
        for provider in providers:
            filename = f'test_{task}_skills_{provider}.csv'
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                skills = df[df['is_skill'] == True]
                all_skills[provider] = {
                    'count': len(skills),
                    'categories': skills['skill_category'].value_counts().to_dict(),
                    'sample_skills': skills['term'].head(3).tolist()
                }
        
        # Print comparison
        for provider, data in all_skills.items():
            print(f"\n{provider.upper()}:")
            print(f"  Total skills: {data['count']}")
            print(f"  Categories: {data['categories']}")
            print(f"  Sample skills: {data['sample_skills']}")

if __name__ == "__main__":
    # Run tests
    test_extraction_sample()
    compare_outputs()
