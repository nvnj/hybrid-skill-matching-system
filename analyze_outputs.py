#!/usr/bin/env python3
"""
Analyze existing skill extraction outputs to identify issues and improvements
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

def analyze_existing_outputs():
    """Analyze the existing CSV outputs to identify issues"""
    
    print("ANALYSIS OF EXISTING SKILL EXTRACTION OUTPUTS")
    print("=" * 60)
    
    # Files to analyze
    files = [
        ('job_skills_extracted_deepseek.csv', 'DeepSeek Job Skills'),
        ('job_skills_extracted_llama.csv', 'LLaMA Job Skills'),
        ('job_skills_extracted_gpt-4.csv', 'GPT-4 Job Skills'),
        ('course_skills_extracted_deepseek.csv', 'DeepSeek Course Skills'),
        ('course_skills_extracted_llama.csv', 'LLaMA Course Skills'),
        ('course_skills_extracted_gpt-4.csv', 'GPT-4 Course Skills')
    ]
    
    all_analyses = {}
    
    for filename, description in files:
        try:
            df = pd.read_csv(filename)
            analysis = analyze_single_file(df, description)
            all_analyses[filename] = analysis
            print(f"\n{description}")
            print("-" * len(description))
            print_analysis(analysis)
            
        except FileNotFoundError:
            print(f"\nFile not found: {filename}")
            continue
        except Exception as e:
            print(f"\nError analyzing {filename}: {e}")
            continue
    
    # Compare across models
    print("\n" + "=" * 60)
    print("CROSS-MODEL COMPARISON")
    print("=" * 60)
    compare_models(all_analyses)

def analyze_single_file(df, description):
    """Analyze a single CSV file"""
    
    analysis = {
        'total_entries': len(df),
        'total_skills': len(df[df['is_skill'] == True]),
        'total_non_skills': len(df[df['is_skill'] == False]),
        'missing_categories': 0,
        'formatting_issues': [],
        'category_distribution': {},
        'duplicate_terms': [],
        'common_skills': [],
        'common_non_skills': []
    }
    
    # Skills analysis
    skills_df = df[df['is_skill'] == True]
    if len(skills_df) > 0:
        # Missing categories
        analysis['missing_categories'] = skills_df['skill_category'].isna().sum()
        
        # Category distribution
        analysis['category_distribution'] = skills_df['skill_category'].value_counts().to_dict()
        
        # Common skills
        analysis['common_skills'] = skills_df['term'].value_counts().head(10).to_dict()
    
    # Non-skills analysis
    non_skills_df = df[df['is_skill'] == False]
    if len(non_skills_df) > 0:
        analysis['common_non_skills'] = non_skills_df['term'].value_counts().head(5).to_dict()
    
    # Formatting issues
    for _, row in df.iterrows():
        term = str(row['term'])
        
        # Check for ** formatting
        if '**' in term:
            analysis['formatting_issues'].append(f"Asterisk formatting: {term}")
        
        # Check for empty terms
        if pd.isna(term) or term.strip() == '':
            analysis['formatting_issues'].append("Empty term found")
        
        # Check for malformed reasoning
        reasoning = str(row['reasoning'])
        if reasoning.endswith(',,') or reasoning.count(',') > 3:
            analysis['formatting_issues'].append(f"Malformed reasoning: {reasoning[:50]}...")
    
    # Check for duplicates
    term_counts = df['term'].value_counts()
    duplicates = term_counts[term_counts > 1]
    if len(duplicates) > 0:
        analysis['duplicate_terms'] = duplicates.head(5).to_dict()
    
    return analysis

def print_analysis(analysis):
    """Print the analysis results"""
    
    print(f"Total entries: {analysis['total_entries']}")
    print(f"Skills: {analysis['total_skills']} ({analysis['total_skills']/analysis['total_entries']*100:.1f}%)")
    print(f"Non-skills: {analysis['total_non_skills']} ({analysis['total_non_skills']/analysis['total_entries']*100:.1f}%)")
    
    if analysis['missing_categories'] > 0:
        print(f"⚠️  Missing skill categories: {analysis['missing_categories']}")
    
    if analysis['formatting_issues']:
        print(f"⚠️  Formatting issues found: {len(analysis['formatting_issues'])}")
        for issue in analysis['formatting_issues'][:3]:  # Show first 3
            print(f"    - {issue}")
        if len(analysis['formatting_issues']) > 3:
            print(f"    ... and {len(analysis['formatting_issues']) - 3} more")
    
    if analysis['duplicate_terms']:
        print(f"⚠️  Duplicate terms found:")
        for term, count in list(analysis['duplicate_terms'].items())[:3]:
            print(f"    - {term}: {count} times")
    
    if analysis['category_distribution']:
        print("Skill categories:")
        for category, count in analysis['category_distribution'].items():
            if category:  # Skip None
                print(f"    {category}: {count}")
    
    print("Top skills:")
    for skill, count in list(analysis['common_skills'].items())[:5]:
        print(f"    {skill}: {count}")

def compare_models(all_analyses):
    """Compare results across different models"""
    
    # Group by type (job vs course)
    job_analyses = {k: v for k, v in all_analyses.items() if 'job_skills' in k}
    course_analyses = {k: v for k, v in all_analyses.items() if 'course_skills' in k}
    
    for task_type, analyses in [('JOB SKILLS', job_analyses), ('COURSE SKILLS', course_analyses)]:
        if not analyses:
            continue
            
        print(f"\n{task_type} COMPARISON:")
        print("-" * 30)
        
        # Extract model names
        model_data = {}
        for filename, analysis in analyses.items():
            if 'deepseek' in filename:
                model_data['DeepSeek'] = analysis
            elif 'llama' in filename:
                model_data['LLaMA'] = analysis
            elif 'gpt-4' in filename:
                model_data['GPT-4'] = analysis
        
        # Compare metrics
        print("Model Performance:")
        for model, data in model_data.items():
            skill_rate = data['total_skills'] / data['total_entries'] * 100 if data['total_entries'] > 0 else 0
            print(f"  {model}: {data['total_skills']} skills from {data['total_entries']} entries ({skill_rate:.1f}%)")
            if data['missing_categories'] > 0:
                print(f"    ⚠️  Missing categories: {data['missing_categories']}")
        
        # Find common skills across models
        all_skills = set()
        for model, data in model_data.items():
            all_skills.update(data['common_skills'].keys())
        
        print(f"\nUnique skills identified: {len(all_skills)}")
        
        # Check consistency
        print("Category usage:")
        all_categories = set()
        for model, data in model_data.items():
            all_categories.update(data['category_distribution'].keys())
        
        for category in sorted(all_categories):
            if category:  # Skip None
                counts = []
                for model, data in model_data.items():
                    count = data['category_distribution'].get(category, 0)
                    counts.append(f"{model}: {count}")
                print(f"  {category}: {', '.join(counts)}")

def generate_recommendations():
    """Generate recommendations based on the analysis"""
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("=" * 60)
    
    recommendations = [
        "1. Fix formatting issues:",
        "   - Remove ** formatting from DeepSeek outputs",
        "   - Ensure consistent term extraction",
        "   - Clean malformed reasoning text",
        "",
        "2. Improve category extraction:",
        "   - Fix parsing to always extract categories from reasoning",
        "   - Add fallback category assignment for missing categories",
        "   - Standardize category names across models",
        "",
        "3. Handle duplicates:",
        "   - Implement deduplication logic",
        "   - Case-insensitive term matching",
        "   - Remove terms with different formatting but same meaning",
        "",
        "4. Model-specific improvements:",
        "   - Add specific system prompts for each model",
        "   - Adjust temperature and parameters for consistency",
        "   - Add validation and post-processing steps",
        "",
        "5. Output standardization:",
        "   - Ensure all models follow the same output format",
        "   - Add validation checks before saving",
        "   - Implement quality scoring for extracted skills"
    ]
    
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    analyze_existing_outputs()
    generate_recommendations()
