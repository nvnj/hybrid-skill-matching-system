import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import re
from dataclasses import dataclass
from enum import Enum
import time
import os
from tqdm import tqdm
import logging
from datetime import datetime


# Azure SDK imports
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from authentication import AZURE_GPT4_API_KEY, AZURE_DEEPSEEK_API_KEY, AZURE_LLAMA_API_KEY

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Track performance metrics for model evaluation"""
    model_name: str
    start_time: float
    end_time: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_skills_extracted: int = 0
    total_non_skills_extracted: int = 0
    avg_response_time: float = 0.0
    request_times: List[float] = None
    error_details: List[str] = None
    
    def __post_init__(self):
        if self.request_times is None:
            self.request_times = []
        if self.error_details is None:
            self.error_details = []
    
    def add_request_time(self, response_time: float):
        """Add a request response time"""
        self.request_times.append(response_time)
        self.avg_response_time = sum(self.request_times) / len(self.request_times)
    
    def add_successful_request(self, response_time: float, skills_count: int, non_skills_count: int, tokens_used: int = 0):
        """Record a successful request"""
        self.successful_requests += 1
        self.total_requests += 1
        self.total_skills_extracted += skills_count
        self.total_non_skills_extracted += non_skills_count
        self.total_tokens_used += tokens_used
        self.add_request_time(response_time)
    
    def add_failed_request(self, error_message: str):
        """Record a failed request"""
        self.failed_requests += 1
        self.total_requests += 1
        self.error_details.append(error_message)
    
    def finalize(self):
        """Finalize metrics calculation"""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        total_time = self.end_time - self.start_time
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            'model_name': self.model_name,
            'total_time_seconds': round(total_time, 2),
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate_percent': round(success_rate, 2),
            'avg_response_time_seconds': round(self.avg_response_time, 2),
            'total_skills_extracted': self.total_skills_extracted,
            'total_non_skills_extracted': self.total_non_skills_extracted,
            'skills_per_minute': round((self.total_skills_extracted / (total_time / 60)), 2) if total_time > 0 else 0,
            'total_tokens_used': self.total_tokens_used,
            'tokens_per_minute': round((self.total_tokens_used / (total_time / 60)), 2) if total_time > 0 else 0
        }

class PerformanceMonitor:
    """Monitor and track performance across models"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.session_start_time = time.time()
        
        if self.enabled:
            logger.info("Performance monitoring enabled")
    
    def start_model_tracking(self, model_name: str):
        """Start tracking for a specific model"""
        if not self.enabled:
            return
        
        self.metrics[model_name] = PerformanceMetrics(
            model_name=model_name,
            start_time=time.time()
        )
        logger.info(f"Started performance tracking for {model_name}")
    
    def record_request(self, model_name: str, success: bool, response_time: float, 
                      skills_count: int = 0, non_skills_count: int = 0, 
                      tokens_used: int = 0, error_message: str = None):
        """Record a request result"""
        if not self.enabled or model_name not in self.metrics:
            return
        
        if success:
            self.metrics[model_name].add_successful_request(
                response_time, skills_count, non_skills_count, tokens_used
            )
        else:
            self.metrics[model_name].add_failed_request(error_message or "Unknown error")
    
    def finalize_model_tracking(self, model_name: str):
        """Finalize tracking for a specific model"""
        if not self.enabled or model_name not in self.metrics:
            return
        
        self.metrics[model_name].finalize()
        logger.info(f"Finalized performance tracking for {model_name}")
    
    def get_model_summary(self, model_name: str) -> Dict:
        """Get summary for a specific model"""
        if not self.enabled or model_name not in self.metrics:
            return {}
        
        return self.metrics[model_name].get_summary()
    
    def get_comparative_report(self) -> Dict:
        """Get comparative report across all models"""
        if not self.enabled:
            return {"performance_monitoring": "disabled"}
        
        report = {
            "session_start": datetime.fromtimestamp(self.session_start_time).isoformat(),
            "session_duration_minutes": round((time.time() - self.session_start_time) / 60, 2),
            "models": {}
        }
        
        for model_name, metrics in self.metrics.items():
            report["models"][model_name] = metrics.get_summary()
        
        # Add comparative metrics
        if len(self.metrics) > 1:
            report["comparison"] = self._generate_comparison()
        
        return report
    
    def _generate_comparison(self) -> Dict:
        """Generate comparison metrics between models"""
        summaries = [metrics.get_summary() for metrics in self.metrics.values()]
        
        # Find best performing model by different metrics
        best_success_rate = max(summaries, key=lambda x: x['success_rate_percent'])
        fastest_avg_response = min(summaries, key=lambda x: x['avg_response_time_seconds'])
        most_skills_extracted = max(summaries, key=lambda x: x['total_skills_extracted'])
        
        return {
            "best_success_rate": {
                "model": best_success_rate['model_name'],
                "rate": best_success_rate['success_rate_percent']
            },
            "fastest_response": {
                "model": fastest_avg_response['model_name'],
                "time": fastest_avg_response['avg_response_time_seconds']
            },
            "most_productive": {
                "model": most_skills_extracted['model_name'],
                "skills_count": most_skills_extracted['total_skills_extracted']
            }
        }
    
    def save_report(self, filename: str = None):
        """Save performance report to file"""
        if not self.enabled:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report = self.get_comparative_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {filename}")
        return filename
    
    def print_summary(self):
        """Print a summary to console"""
        if not self.enabled:
            print("Performance monitoring is disabled")
            return
        
        report = self.get_comparative_report()
        
        print("\n" + "="*60)
        print("PERFORMANCE MONITORING SUMMARY")
        print("="*60)
        
        print(f"Session Duration: {report['session_duration_minutes']} minutes")
        print(f"Models Tested: {len(report['models'])}")
        
        for model_name, metrics in report['models'].items():
            print(f"\n{model_name.upper()} PERFORMANCE:")
            print(f"  Success Rate: {metrics['success_rate_percent']}%")
            print(f"  Avg Response Time: {metrics['avg_response_time_seconds']}s")
            print(f"  Skills Extracted: {metrics['total_skills_extracted']}")
            print(f"  Skills/Minute: {metrics['skills_per_minute']}")
            print(f"  Total Requests: {metrics['total_requests']}")
            if metrics['total_tokens_used'] > 0:
                print(f"  Tokens Used: {metrics['total_tokens_used']}")
        
        if 'comparison' in report:
            print(f"\nCOMPARATIVE RESULTS:")
            comp = report['comparison']
            print(f"  Best Success Rate: {comp['best_success_rate']['model']} ({comp['best_success_rate']['rate']}%)")
            print(f"  Fastest Response: {comp['fastest_response']['model']} ({comp['fastest_response']['time']}s)")
            print(f"  Most Productive: {comp['most_productive']['model']} ({comp['most_productive']['skills_count']} skills)")

class LLMProvider(Enum):
    """Supported LLM providers via Azure"""
    GPT4 = "gpt-4"
    DEEPSEEK = "deepseek"
    LLAMA = "llama"

@dataclass
class AzureModelConfig:
    """Configuration for Azure models"""
    # GPT-4 configuration (Azure OpenAI)
    gpt4_endpoint: str = "https://naveenjohn-5656-resource.cognitiveservices.azure.com/"
    gpt4_deployment: str = "gpt-4o"
    gpt4_api_version: str = "2024-12-01-preview"
    
    # DeepSeek configuration (Azure AI Inference)
    deepseek_endpoint: str = "https://naveenjohn-5656-resource.services.ai.azure.com/models"
    deepseek_model: str = "DeepSeek-R1-0528"
    deepseek_api_version: str = "2024-05-01-preview"
    
    # LLaMA configuration (Azure AI Inference)
    llama_endpoint: str = "https://naveenjohn-5656-resource.services.ai.azure.com/models"
    llama_model: str = "Llama-3.3-70B-Instruct"
    llama_api_version: str = "2024-05-01-preview"

@dataclass
class JobSkillExtractionPrompt:
    """Prompt template for job description skill extraction following PromptNER methodology"""
    
    definition: str = """
    Defn: A job skill is a technical_skill(programming language, framework, tool, platform, technology), 
    soft_skill(communication, leadership, teamwork, problem-solving), 
    domain_knowledge(industry expertise, methodology, business concept),
    certification(degree, license, professional qualification),
    analytical_skill(data analysis, research, quantitative methods), or
    operational_skill(maintenance, inspection, safety procedures, equipment operation).
    Skills are specific competencies that can be learned and applied in a workplace.
    Company names, job titles, locations, generic business terms, responsibilities without specific skills, 
    dates, numbers, and vague descriptive phrases are NOT skills.
    
    IMPORTANT: If you identify a skill that doesn't fit the predefined categories, you may create a new appropriate category.
    Always include the skill category in parentheses at the end of your reasoning.
    """
    
    few_shot_examples: str = """
    Example 1: Looking for a Senior Software Engineer with 5+ years experience in Java, Spring Boot, 
    and microservices architecture. Must have strong knowledge of AWS, Docker, Kubernetes, and CI/CD pipelines. 
    Experience with Agile methodology and excellent communication skills required. Position based in San Francisco.
    
    Answer:
    1. Java | True | it is a programming language used for software development (technical_skill)
    2. Spring Boot | True | it is a Java framework for building applications (technical_skill)
    3. microservices architecture | True | it is a software design pattern and methodology (domain_knowledge)
    4. AWS | True | it is a cloud computing platform (technical_skill)
    5. Docker | True | it is a containerization platform (technical_skill)
    6. Kubernetes | True | it is a container orchestration platform (technical_skill)
    7. CI/CD pipelines | True | it is a software development practice (domain_knowledge)
    8. Agile methodology | True | it is a project management framework (domain_knowledge)
    9. communication skills | True | it is an interpersonal soft skill (soft_skill)
    10. Senior Software Engineer | False | it is a job title, not a skill
    11. 5+ years experience | False | it is a time requirement, not a skill
    12. San Francisco | False | it is a location, not a skill
    13. Position | False | it is a generic term, not a skill
    
    Example 2: Data Analyst role requiring proficiency in SQL, Python, and Tableau. 
    Must understand statistical analysis, A/B testing, and data visualization. 
    Bachelor's degree in Statistics or related field required. Report to Analytics Manager.
    
    Answer:
    1. SQL | True | it is a database query language (technical_skill)
    2. Python | True | it is a programming language (technical_skill)
    3. Tableau | True | it is a data visualization tool (technical_skill)
    4. statistical analysis | True | it is an analytical methodology (analytical_skill)
    5. A/B testing | True | it is an experimental design methodology (analytical_skill)
    6. data visualization | True | it is a skill for presenting data graphically (technical_skill)
    7. Bachelor's degree in Statistics | True | it is an educational qualification (certification)
    8. Data Analyst | False | it is a job title, not a skill
    9. Analytics Manager | False | it is a job title/reporting structure, not a skill
    10. Report to | False | it is an organizational structure term, not a skill
    
    FORMAT REQUIREMENTS:
    - Follow the exact format: Number. Term | True/False | reasoning (category)
    - Do not use ** or other formatting around terms
    - Always include the skill category in parentheses at the end of reasoning for skills
    - Be consistent and avoid duplicates
    """
    
    def create_prompt(self, job_description: str) -> str:
        """Create the full prompt for skill extraction from job description"""
        return f"""
{self.definition}

{self.few_shot_examples}

Q: Given the job description below, identify all possible skills and for each entry explain why it either is or is not a skill.

IMPORTANT INSTRUCTIONS:
1. Follow the exact format: Number. Term | True/False | reasoning (category)
2. Do not use ** or other formatting around terms
3. Always include the skill category in parentheses for skills
4. Avoid duplicates - each skill should appear only once
5. If a skill doesn't fit predefined categories, create an appropriate new category

Job Description: {job_description}

Answer:
"""

@dataclass
class CourseSkillExtractionPrompt:
    """Prompt template for course learning outcome skill extraction following PromptNER methodology"""
    
    definition: str = """
    Defn: A course skill is a technical_skill(programming language, software, tool, technology), 
    theoretical_knowledge(concepts, principles, theories, models),
    analytical_skill(analysis methods, problem-solving techniques, research methods),
    practical_skill(hands-on abilities, laboratory techniques, applied methods),
    professional_skill(communication, presentation, teamwork, project management), or
    academic_skill(writing, research, critical thinking, evaluation methods).
    Skills are specific competencies that students learn and can apply.
    Course names, instructor names, university names, generic academic terms, administrative details,
    and vague learning objectives without specific skills are NOT skills.
    
    IMPORTANT: If you identify a skill that doesn't fit the predefined categories, you may create a new appropriate category.
    Always include the skill category in parentheses at the end of your reasoning.
    """
    
    few_shot_examples: str = """
    Example 1: This course covers Python programming fundamentals including data structures, algorithms, 
    and object-oriented programming. Students will learn NumPy, Pandas for data manipulation, and 
    matplotlib for visualization. Emphasis on problem-solving and debugging techniques. 
    Final project involves building a web application using Django.
    
    Answer:
    1. Python programming | True | it is a programming language skill (technical_skill)
    2. data structures | True | it is a fundamental computer science concept (theoretical_knowledge)
    3. algorithms | True | it is a computational problem-solving methodology (theoretical_knowledge)
    4. object-oriented programming | True | it is a programming paradigm (theoretical_knowledge)
    5. NumPy | True | it is a Python library for numerical computing (technical_skill)
    6. Pandas | True | it is a Python library for data manipulation (technical_skill)
    7. data manipulation | True | it is a practical data handling skill (practical_skill)
    8. matplotlib | True | it is a Python visualization library (technical_skill)
    9. visualization | True | it is a skill for presenting data graphically (practical_skill)
    10. problem-solving | True | it is an analytical thinking skill (analytical_skill)
    11. debugging techniques | True | it is a software development skill (practical_skill)
    12. Django | True | it is a Python web framework (technical_skill)
    13. web application | True | it is a type of software development skill (practical_skill)
    14. This course | False | it is a reference phrase, not a skill
    15. Students will learn | False | it is an instructional phrase, not a skill
    16. Final project | False | it is an assessment method, not a skill
    
    Example 2: Introduction to Machine Learning covering supervised and unsupervised learning, 
    neural networks, and deep learning frameworks. Practical experience with TensorFlow and scikit-learn. 
    Topics include regression analysis, classification, clustering, and model evaluation. 
    Prerequisites: Linear algebra and statistics.
    
    Answer:
    1. Machine Learning | True | it is a field of artificial intelligence (theoretical_knowledge)
    2. supervised learning | True | it is a machine learning paradigm (theoretical_knowledge)
    3. unsupervised learning | True | it is a machine learning paradigm (theoretical_knowledge)
    4. neural networks | True | it is a machine learning architecture (theoretical_knowledge)
    5. deep learning | True | it is an advanced machine learning technique (theoretical_knowledge)
    6. TensorFlow | True | it is a machine learning framework (technical_skill)
    7. scikit-learn | True | it is a machine learning library (technical_skill)
    8. regression analysis | True | it is a statistical modeling technique (analytical_skill)
    9. classification | True | it is a machine learning task (analytical_skill)
    10. clustering | True | it is an unsupervised learning technique (analytical_skill)
    11. model evaluation | True | it is a machine learning validation skill (analytical_skill)
    12. Linear algebra | True | it is mathematical foundation knowledge (theoretical_knowledge)
    13. statistics | True | it is mathematical/analytical knowledge (theoretical_knowledge)
    14. Introduction to | False | it is a course title prefix, not a skill
    15. Prerequisites | False | it is an administrative term, not a skill
    16. Topics include | False | it is a descriptive phrase, not a skill
    
    FORMAT REQUIREMENTS:
    - Follow the exact format: Number. Term | True/False | reasoning (category)
    - Do not use ** or other formatting around terms
    - Always include the skill category in parentheses at the end of reasoning for skills
    - Be consistent and avoid duplicates
    """
    
    def create_prompt(self, course_description: str) -> str:
        """Create the full prompt for skill extraction from course learning outcomes"""
        return f"""
{self.definition}

{self.few_shot_examples}

Q: Given the course learning outcomes below, identify all possible skills and for each entry explain why it either is or is not a skill.

IMPORTANT INSTRUCTIONS:
1. Follow the exact format: Number. Term | True/False | reasoning (category)
2. Do not use ** or other formatting around terms
3. Always include the skill category in parentheses for skills
4. Avoid duplicates - each skill should appear only once
5. If a skill doesn't fit predefined categories, create an appropriate new category

Course Learning Outcomes: {course_description}

Answer:
"""

class AzureSkillExtractor:
    """Extract skills using Azure OpenAI and Azure AI Inference"""
    
    def __init__(self, 
                 provider: LLMProvider = LLMProvider.GPT4,
                 api_key: Optional[str] = None,
                 performance_monitor: PerformanceMonitor = None):
        
        self.provider = provider
        self.config = AzureModelConfig()
        self.job_prompt_template = JobSkillExtractionPrompt()
        self.course_prompt_template = CourseSkillExtractionPrompt()
        self.performance_monitor = performance_monitor
        
        # Model-specific system messages
        self.system_messages = {
            LLMProvider.GPT4: "You are an expert at identifying skills and competencies. Follow the exact format provided in the examples. Be precise and consistent. Always include skill categories in parentheses.",
            LLMProvider.DEEPSEEK: "You are an expert at identifying skills and competencies. CRITICAL INSTRUCTIONS: 1) Follow the exact format: Number. Term | True/False | reasoning (category) 2) Do NOT use ** or any special formatting around terms 3) Always include skill category in parentheses at the end of reasoning 4) Avoid duplicates - each skill should appear only once 5) Be consistent and precise",
            LLMProvider.LLAMA: "You are an expert at identifying skills and competencies. IMPORTANT INSTRUCTIONS: 1) Follow the exact format: Number. Term | True/False | reasoning (category) 2) Always end your reasoning with the skill category in parentheses like (technical_skill) 3) Do not repeat skills 4) Be consistent and precise 5) Include skill category for every skill identified"
        }
        
        # Initialize the appropriate client based on provider
        if provider == LLMProvider.GPT4:
            # Use Azure OpenAI client for GPT-4
            self.api_key = AZURE_GPT4_API_KEY
            if not self.api_key:
                raise ValueError("GPT-4 API key not found. Set AZURE_GPT4_API_KEY environment variable.")
            
            self.client = AzureOpenAI(
                api_version=self.config.gpt4_api_version,
                azure_endpoint=self.config.gpt4_endpoint,
                api_key=self.api_key
            )
            self.model_name = self.config.gpt4_deployment
            
        elif provider == LLMProvider.DEEPSEEK:
            # Use Azure AI Inference client for DeepSeek
            self.api_key = AZURE_DEEPSEEK_API_KEY
            if not self.api_key:
                raise ValueError("DeepSeek API key not found. Set AZURE_DEEPSEEK_API_KEY environment variable.")
            
            self.client = ChatCompletionsClient(
                endpoint=self.config.deepseek_endpoint,
                credential=AzureKeyCredential(self.api_key),
                api_version=self.config.deepseek_api_version
            )
            self.model_name = self.config.deepseek_model
            
        elif provider == LLMProvider.LLAMA:
            # Use Azure AI Inference client for LLaMA
            self.api_key = AZURE_LLAMA_API_KEY
            if not self.api_key:
                raise ValueError("LLaMA API key not found. Set AZURE_LLAMA_API_KEY environment variable.")
            
            self.client = ChatCompletionsClient(
                endpoint=self.config.llama_endpoint,
                credential=AzureKeyCredential(self.api_key),
                api_version=self.config.llama_api_version
            )
            self.model_name = self.config.llama_model
        
        logger.info(f"Initialized Azure {provider.value} extractor with model: {self.model_name}")
    
    def extract_skills_with_reasoning(self, text: str, prompt_template, max_retries: int = 3) -> List[Dict]:
        """
        Extract skills with reasoning using Azure services
        
        Returns:
            List of dictionaries with format:
            {
                'term': 'Python',
                'is_skill': True,
                'reasoning': 'it is a programming language (technical_skill)',
                'skill_category': 'technical_skill'
            }
        """
        prompt = prompt_template.create_prompt(text)
        request_start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                if self.provider == LLMProvider.GPT4:
                    response_text, tokens_used = self._call_azure_openai(prompt)
                else:  # DeepSeek or LLaMA
                    response_text, tokens_used = self._call_azure_ai_inference(prompt)
                
                request_time = time.time() - request_start_time
                skills_with_reasoning = self._parse_skill_response(response_text)
                skills_with_reasoning = self._validate_and_clean_skills(skills_with_reasoning)
                
                # Record performance metrics
                if self.performance_monitor:
                    skills_count = len([s for s in skills_with_reasoning if s['is_skill']])
                    non_skills_count = len(skills_with_reasoning) - skills_count
                    self.performance_monitor.record_request(
                        self.model_name, True, request_time, 
                        skills_count, non_skills_count, tokens_used
                    )
                
                return skills_with_reasoning
                
            except Exception as e:
                request_time = time.time() - request_start_time
                error_msg = str(e)
                logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
                
                # Record failed request
                if self.performance_monitor:
                    self.performance_monitor.record_request(
                        self.model_name, False, request_time, 
                        error_message=error_msg
                    )
                
                if attempt == max_retries - 1:
                    logger.error(f"All attempts failed for text: {text[:100]}...")
                    return []
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return []
    
    def _call_azure_openai(self, prompt: str) -> Tuple[str, int]:
        """Call Azure OpenAI for GPT-4"""
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.system_messages[self.provider]
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2000,
            temperature=0.1,
            top_p=1.0,
            model=self.model_name
        )
        
        # Extract token usage if available
        tokens_used = 0
        if hasattr(response, 'usage') and response.usage:
            tokens_used = response.usage.total_tokens
        
        return response.choices[0].message.content, tokens_used
    
    def _call_azure_ai_inference(self, prompt: str) -> Tuple[str, int]:
        """Call Azure AI Inference for DeepSeek or LLaMA"""
        response = self.client.complete(
            messages=[
                SystemMessage(content=self.system_messages[self.provider]),
                UserMessage(content=prompt)
            ],
            max_tokens=2000,
            temperature=0.1,
            top_p=0.9,
            model=self.model_name
        )
        
        # Extract token usage if available
        tokens_used = 0
        if hasattr(response, 'usage') and response.usage:
            tokens_used = response.usage.total_tokens
        
        return response.choices[0].message.content, tokens_used
    
    def _parse_skill_response(self, response_text: str) -> List[Dict]:
        """Parse the LLM response to extract skills with reasoning"""
        skills_data = []
        lines = response_text.strip().split('\n')
        seen_terms = set()  # Track seen terms to avoid duplicates
        
        for line in lines:
            line = line.strip()
            if not line or line.lower().startswith(('answer:', 'q:', 'question:')):
                continue
                
            # Handle different formats: | or separated by spaces
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    # Extract components
                    term_part = parts[0].strip()
                    is_skill_part = parts[1].strip()
                    reasoning_part = parts[2].strip() if len(parts) > 2 else ""
                    
                    # Clean term (remove numbering and ** formatting)
                    term = re.sub(r'^\d+\.\s*', '', term_part).strip()
                    # Remove various formatting: **, *, bold, etc.
                    term = re.sub(r'\*\*([^*]+)\*\*', r'\1', term)  # Remove ** formatting
                    term = re.sub(r'\*([^*]+)\*', r'\1', term)      # Remove * formatting
                    term = re.sub(r'__([^_]+)__', r'\1', term)      # Remove __ formatting
                    term = re.sub(r'`([^`]+)`', r'\1', term)        # Remove ` formatting
                    term = term.strip()
                    
                    # Normalize term for duplicate detection
                    term_normalized = re.sub(r'\s+', ' ', term.lower().strip())
                    
                    # Skip if we've seen this term already (normalized comparison)
                    if term_normalized in seen_terms or not term_normalized:
                        logger.debug(f"Skipping duplicate or empty term: '{term}'")
                        continue
                    seen_terms.add(term_normalized)
                    
                    # Determine if it's a skill
                    is_skill = 'True' in is_skill_part or 'true' in is_skill_part.lower()
                    
                    # Extract skill category from reasoning
                    skill_category = None
                    if is_skill and reasoning_part:
                        # Look for category in parentheses at the end
                        category_match = re.search(r'\(([^)]+)\)(?:\s*$)', reasoning_part)
                        if category_match:
                            skill_category = category_match.group(1).strip()
                        else:
                            # If no category found but it's a skill, try to infer from reasoning
                            reasoning_lower = reasoning_part.lower()
                            if 'technical' in reasoning_lower or 'programming' in reasoning_lower or 'software' in reasoning_lower or 'tool' in reasoning_lower:
                                skill_category = 'technical_skill'
                            elif 'communication' in reasoning_lower or 'leadership' in reasoning_lower or 'teamwork' in reasoning_lower or 'interpersonal' in reasoning_lower:
                                skill_category = 'soft_skill'
                            elif 'domain' in reasoning_lower or 'industry' in reasoning_lower or 'methodology' in reasoning_lower or 'business' in reasoning_lower:
                                skill_category = 'domain_knowledge'
                            elif 'certification' in reasoning_lower or 'degree' in reasoning_lower or 'license' in reasoning_lower or 'qualification' in reasoning_lower:
                                skill_category = 'certification'
                            elif 'analytical' in reasoning_lower or 'analysis' in reasoning_lower or 'research' in reasoning_lower or 'data' in reasoning_lower:
                                skill_category = 'analytical_skill'
                            elif 'maintenance' in reasoning_lower or 'operation' in reasoning_lower or 'inspection' in reasoning_lower:
                                skill_category = 'operational_skill'
                            else:
                                skill_category = 'unclassified'
                    
                    # Clean reasoning (remove extra quotes and formatting)
                    reasoning_part = re.sub(r'^["\']|["\']$', '', reasoning_part)
                    reasoning_part = reasoning_part.strip()
                    
                    if term and not term.lower() in ['answer:', 'question:', 'q:', '']:
                        skills_data.append({
                            'term': term,
                            'is_skill': is_skill,
                            'reasoning': reasoning_part,
                            'skill_category': skill_category
                        })
        
        logger.info(f"Parsed {len(skills_data)} unique skills from response ({self.provider.value})")
        return skills_data
    
    def _validate_and_clean_skills(self, skills_data: List[Dict]) -> List[Dict]:
        """Validate and clean the extracted skills data"""
        cleaned_skills = []
        seen_terms_normalized = set()
        
        for skill_item in skills_data:
            # Skip empty terms
            if not skill_item.get('term') or skill_item['term'].strip() == '':
                continue
            
            # Clean term
            term = skill_item['term'].strip()
            
            # Additional cleaning for any remaining formatting issues
            term = re.sub(r'\*\*([^*]+)\*\*', r'\1', term)  # Remove ** formatting
            term = re.sub(r'\*([^*]+)\*', r'\1', term)      # Remove * formatting  
            term = re.sub(r'__([^_]+)__', r'\1', term)      # Remove __ formatting
            term = re.sub(r'`([^`]+)`', r'\1', term)        # Remove ` formatting
            term = term.strip()
            
            # Normalize for duplicate checking
            term_normalized = re.sub(r'\s+', ' ', term.lower().strip())
            
            # Skip duplicates
            if term_normalized in seen_terms_normalized:
                logger.debug(f"Skipping duplicate in validation: '{term}'")
                continue
            seen_terms_normalized.add(term_normalized)
            
            # Skip if term is too generic or clearly not a skill
            skip_terms = ['answer:', 'question:', 'q:', 'example', 'note:', 'format requirements', 'instructions:']
            if any(skip_term in term.lower() for skip_term in skip_terms):
                continue
            
            # Update the cleaned term
            skill_item['term'] = term
            
            # Ensure skill_category is set for skills
            if skill_item.get('is_skill') and not skill_item.get('skill_category'):
                # Try to infer category from reasoning
                reasoning = skill_item.get('reasoning', '').lower()
                if 'technical' in reasoning or 'programming' in reasoning or 'software' in reasoning or 'tool' in reasoning:
                    skill_item['skill_category'] = 'technical_skill'
                elif 'soft' in reasoning or 'communication' in reasoning or 'leadership' in reasoning or 'interpersonal' in reasoning:
                    skill_item['skill_category'] = 'soft_skill'
                elif 'domain' in reasoning or 'methodology' in reasoning or 'industry' in reasoning or 'business' in reasoning:
                    skill_item['skill_category'] = 'domain_knowledge'
                elif 'certification' in reasoning or 'degree' in reasoning or 'license' in reasoning or 'qualification' in reasoning:
                    skill_item['skill_category'] = 'certification'
                elif 'analytical' in reasoning or 'analysis' in reasoning or 'research' in reasoning or 'data' in reasoning:
                    skill_item['skill_category'] = 'analytical_skill'
                elif 'maintenance' in reasoning or 'operation' in reasoning or 'inspection' in reasoning or 'safety' in reasoning:
                    skill_item['skill_category'] = 'operational_skill'
                else:
                    skill_item['skill_category'] = 'unclassified'
                    
                logger.debug(f"Inferred category '{skill_item['skill_category']}' for skill '{term}'")
            
            cleaned_skills.append(skill_item)
        
        logger.info(f"Validation completed: {len(cleaned_skills)} valid skills after cleaning")
        return cleaned_skills

class BatchSkillProcessor:
    """Process CSV files to extract skills with reasoning using Azure models"""
    
    def __init__(self, 
                 provider: LLMProvider = LLMProvider.GPT4,
                 api_key: Optional[str] = None,
                 rate_limit_delay: float = 1.0,
                 enable_performance_monitoring: bool = True):
        
        self.provider = provider
        self.rate_limit_delay = rate_limit_delay
        self.job_results = []
        self.course_results = []
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor(enabled=enable_performance_monitoring)
        
        # Initialize extractor with performance monitor
        self.extractor = AzureSkillExtractor(
            provider=provider, 
            api_key=api_key,
            performance_monitor=self.performance_monitor
        )
        
        logger.info(f"Initialized batch processor with Azure {provider.value} model: {self.extractor.model_name}")
        if enable_performance_monitoring:
            logger.info("Performance monitoring enabled")
            self.performance_monitor.start_model_tracking(self.extractor.model_name)
    
    def process_jobs_csv(self, 
                        input_file: str, 
                        output_file: str = None,
                        sample_size: int = None) -> pd.DataFrame:
        """
        Process jobs CSV to extract skills with reasoning
        
        Args:
            input_file: Path to Jobs.csv
            output_file: Path for output CSV (auto-generated if not provided)
            sample_size: Optional number of rows to process (for testing)
        """
        if output_file is None:
            output_file = f"job_skills_extracted_{self.provider.value}.csv"
            
        logger.info(f"Loading jobs from {input_file}")
        logger.info(f"Using Azure {self.provider.value} model: {self.extractor.model_name}")
        
        # Read CSV with appropriate encoding
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
        except:
            df = pd.read_csv(input_file, encoding='latin-1')
        
        if sample_size:
            df = df.head(sample_size)
        
        logger.info(f"Processing {len(df)} job descriptions")
        
        all_results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing jobs with {self.provider.value}"):
            job_id = row['Job_Id']
            job_description = row['job_description']
            
            # Skip if description is empty or NaN
            if pd.isna(job_description) or str(job_description).strip() == '':
                logger.warning(f"Skipping job {job_id} - empty description")
                continue
            
            # Extract skills with reasoning
            try:
                skills_data = self.extractor.extract_skills_with_reasoning(
                    job_description,
                    self.extractor.job_prompt_template
                )
                
                if not skills_data:
                    logger.warning(f"No skills extracted for job {job_id}")
                    continue
                    
            except Exception as e:
                logger.error(f"Failed to extract skills for job {job_id}: {e}")
                continue
            
            # Create rows for each identified term
            for skill_item in skills_data:
                all_results.append({
                    'job_id': job_id,
                    'term': skill_item['term'],
                    'is_skill': skill_item['is_skill'],
                    'reasoning': skill_item['reasoning'],
                    'skill_category': skill_item['skill_category'],
                    'model_used': self.extractor.model_name
                })
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Save intermediate results every 10 jobs
            if (idx + 1) % 10 == 0:
                temp_df = pd.DataFrame(all_results)
                temp_df.to_csv(f"temp_{output_file}", index=False)
                logger.info(f"Saved intermediate results: {len(all_results)} entries")
        
        # Create final DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved job skills to {output_file}")
        
        # Finalize performance monitoring
        if self.performance_monitor.enabled:
            self.performance_monitor.finalize_model_tracking(self.extractor.model_name)
        
        # Print summary
        self._print_summary(results_df, f"Job Skills ({self.provider.value})")
        
        return results_df
    
    def process_courses_csv(self,
                          input_file: str,
                          output_file: str = None,
                          sample_size: int = None) -> pd.DataFrame:
        """
        Process courses CSV to extract skills with reasoning
        
        Args:
            input_file: Path to Learning_Outcomes_with_Skills.csv
            output_file: Path for output CSV (auto-generated if not provided)
            sample_size: Optional number of rows to process (for testing)
        """
        if output_file is None:
            output_file = f"course_skills_extracted_{self.provider.value}.csv"
            
        logger.info(f"Loading courses from {input_file}")
        logger.info(f"Using Azure {self.provider.value} model: {self.extractor.model_name}")
        
        # Read CSV with appropriate encoding (cp1252 as specified)
        try:
            df = pd.read_csv(input_file, encoding='cp1252')
        except:
            df = pd.read_csv(input_file, encoding='latin-1')
        
        if sample_size:
            df = df.head(sample_size)
        
        logger.info(f"Processing {len(df)} course descriptions")
        
        all_results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing courses with {self.provider.value}"):
            course_code = row['Course_Code']
            combined_text = row['Combined_Text']
            
            # Skip if description is empty or NaN
            if pd.isna(combined_text) or str(combined_text).strip() == '':
                logger.warning(f"Skipping course {course_code} - empty description")
                continue
            
            # Extract skills with reasoning
            try:
                skills_data = self.extractor.extract_skills_with_reasoning(
                    combined_text,
                    self.extractor.course_prompt_template
                )
                
                if not skills_data:
                    logger.warning(f"No skills extracted for course {course_code}")
                    continue
                    
            except Exception as e:
                logger.error(f"Failed to extract skills for course {course_code}: {e}")
                continue
            
            # Create rows for each identified term
            for skill_item in skills_data:
                all_results.append({
                    'course_code': course_code,
                    'term': skill_item['term'],
                    'is_skill': skill_item['is_skill'],
                    'reasoning': skill_item['reasoning'],
                    'skill_category': skill_item['skill_category'],
                    'model_used': self.extractor.model_name
                })
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Save intermediate results every 10 courses
            if (idx + 1) % 10 == 0:
                temp_df = pd.DataFrame(all_results)
                temp_df.to_csv(f"temp_{output_file}", index=False)
                logger.info(f"Saved intermediate results: {len(all_results)} entries")
        
        # Create final DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved course skills to {output_file}")
        
        # Finalize performance monitoring
        if self.performance_monitor.enabled:
            self.performance_monitor.finalize_model_tracking(self.extractor.model_name)
        
        # Print summary
        self._print_summary(results_df, f"Course Skills ({self.provider.value})")
        
        return results_df
    
    def _print_summary(self, df: pd.DataFrame, title: str):
        """Print summary statistics"""
        print(f"\n{'='*60}")
        print(f"{title} Extraction Summary")
        print(f"{'='*60}")
        
        if len(df) > 0:
            # Overall statistics
            total_terms = len(df)
            skills = df[df['is_skill'] == True]
            non_skills = df[df['is_skill'] == False]
            
            print(f"Model Used: {df['model_used'].iloc[0] if 'model_used' in df.columns else 'N/A'}")
            print(f"Total terms analyzed: {total_terms}")
            print(f"Identified as skills: {len(skills)} ({len(skills)/total_terms*100:.1f}%)")
            print(f"Identified as non-skills: {len(non_skills)} ({len(non_skills)/total_terms*100:.1f}%)")
            
            # Category breakdown for skills
            if len(skills) > 0 and 'skill_category' in skills.columns:
                print(f"\nSkill Categories:")
                category_counts = skills['skill_category'].value_counts()
                for category, count in category_counts.items():
                    if category:  # Skip None values
                        print(f"  {category}: {count}")
            
            # Top skills
            if len(skills) > 0:
                print(f"\nTop 10 Most Common Skills:")
                top_skills = skills['term'].value_counts().head(10)
                for skill, count in top_skills.items():
                    print(f"  {skill}: {count}")
            
            # Sample of non-skills
            if len(non_skills) > 0:
                print(f"\nSample of Non-Skills (first 5):")
                for _, row in non_skills.head(5).iterrows():
                    print(f"  {row['term']}: {row['reasoning']}")

def main():
    """Main function to run skill extraction with Azure models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract skills using Azure OpenAI and AI Inference')
    parser.add_argument('--jobs', type=str, help='Path to Jobs.csv file')
    parser.add_argument('--courses', type=str, help='Path to Learning_Outcomes_with_Skills.csv file')
    parser.add_argument('--provider', type=str, default='gpt-4', 
                       choices=['gpt-4', 'deepseek', 'llama'],
                       help='Model provider to use')
    parser.add_argument('--sample', type=int, help='Number of rows to process (for testing)')
    parser.add_argument('--delay', type=float, default=1.0, help='Rate limit delay in seconds')
    parser.add_argument('--all-models', action='store_true', 
                       help='Run extraction with all three models sequentially')
    parser.add_argument('--both', action='store_true',
                       help='Process both jobs and courses (requires --jobs and --courses)')
    parser.add_argument('--api-key', type=str, help='API key for the selected provider')
    parser.add_argument('--no-performance', action='store_true', 
                       help='Disable performance monitoring')
    parser.add_argument('--save-performance', type=str, 
                       help='Save performance report to specified file')
    
    args = parser.parse_args()
    
    # Initialize performance monitoring
    enable_monitoring = not args.no_performance
    global_monitor = PerformanceMonitor(enabled=enable_monitoring)
    
    # Auto-set files if --both is used but files not specified
    if args.both:
        if not args.jobs:
            args.jobs = 'Jobs.csv'
        if not args.courses:
            args.courses = 'Learning_Outcomes_with_Skills.csv'
        print(f"Processing both jobs ({args.jobs}) and courses ({args.courses})")
        if args.sample:
            print(f"Sample size: {args.sample} entries each")
    
    if args.all_models:
        # Run with all three models
        providers = [LLMProvider.GPT4, LLMProvider.DEEPSEEK, LLMProvider.LLAMA]
        
        for provider in providers:
            print(f"\n{'='*60}")
            print(f"Running extraction with Azure {provider.value}")
            print(f"{'='*60}")
            
            try:
                processor = BatchSkillProcessor(
                    provider=provider,
                    api_key=args.api_key,
                    rate_limit_delay=args.delay,
                    enable_performance_monitoring=enable_monitoring
                )
                
                # Process jobs if specified
                if args.jobs:
                    print(f"\nProcessing Job Descriptions with {provider.value}...")
                    job_output = f"job_skills_extracted_{provider.value}.csv"
                    if args.sample:
                        job_output = f"sample_{args.sample}_job_skills_{provider.value}.csv"
                    
                    job_results = processor.process_jobs_csv(
                        args.jobs,
                        output_file=job_output,
                        sample_size=args.sample
                    )
                
                # Process courses if specified
                if args.courses:
                    print(f"\nProcessing Course Learning Outcomes with {provider.value}...")
                    course_output = f"course_skills_extracted_{provider.value}.csv"
                    if args.sample:
                        course_output = f"sample_{args.sample}_course_skills_{provider.value}.csv"
                    
                    course_results = processor.process_courses_csv(
                        args.courses,
                        output_file=course_output,
                        sample_size=args.sample
                    )
                
                # Add model performance to global monitor
                if enable_monitoring:
                    model_summary = processor.performance_monitor.get_model_summary(processor.extractor.model_name)
                    if model_summary:
                        global_monitor.metrics[processor.extractor.model_name] = processor.performance_monitor.metrics[processor.extractor.model_name]
                    
            except Exception as e:
                logger.error(f"Failed to process with {provider.value}: {e}")
                continue
        
        # Print comparative performance report
        if enable_monitoring and len(global_monitor.metrics) > 1:
            global_monitor.print_summary()
            
            # Save performance report if requested
            if args.save_performance:
                global_monitor.save_report(args.save_performance)
            else:
                global_monitor.save_report()  # Auto-generate filename
        
    else:
        # Run with single specified model
        # Map provider argument to correct enum value
        provider_mapping = {
            'gpt-4': LLMProvider.GPT4,
            'deepseek': LLMProvider.DEEPSEEK, 
            'llama': LLMProvider.LLAMA
        }
        
        if args.provider not in provider_mapping:
            print(f"Error: Invalid provider '{args.provider}'. Choose from: {list(provider_mapping.keys())}")
            return
            
        provider = provider_mapping[args.provider]
        
        # Initialize processor
        processor = BatchSkillProcessor(
            provider=provider,
            api_key=args.api_key,
            rate_limit_delay=args.delay,
            enable_performance_monitoring=enable_monitoring
        )
        
        # Process jobs if specified
        if args.jobs:
            print(f"\nProcessing Job Descriptions with Azure {provider.value}...")
            job_output = f"job_skills_extracted_{provider.value}.csv"
            if args.sample:
                job_output = f"sample_{args.sample}_job_skills_{provider.value}.csv"
            
            job_results = processor.process_jobs_csv(
                args.jobs,
                output_file=job_output,
                sample_size=args.sample
            )
        
        # Process courses if specified
        if args.courses:
            print(f"\nProcessing Course Learning Outcomes with Azure {provider.value}...")
            course_output = f"course_skills_extracted_{provider.value}.csv"
            if args.sample:
                course_output = f"sample_{args.sample}_course_skills_{provider.value}.csv"
            
            course_results = processor.process_courses_csv(
                args.courses,
                output_file=course_output,
                sample_size=args.sample
            )
        
        # Print performance report for single model
        if enable_monitoring:
            processor.performance_monitor.print_summary()
            
            # Save performance report if requested
            if args.save_performance:
                processor.performance_monitor.save_report(args.save_performance)
    
    if not args.jobs and not args.courses and not args.both:
        print("Azure Skill Extraction System")
        print("="*60)
        print("\nPlease specify files to process:")
        print("\nExamples:")
        print("  Process first 5 jobs & courses with GPT-4:")
        print("    python3 extraction.py --both --sample 5 --provider gpt-4")
        print("")
        print("  Process first 5 with all models (with performance monitoring):")
        print("    python3 extraction.py --both --sample 5 --all-models")
        print("")
        print("  Process with performance monitoring disabled:")
        print("    python3 extraction.py --both --sample 5 --no-performance")
        print("")
        print("  Save performance report to custom file:")
        print("    python3 extraction.py --both --sample 5 --all-models --save-performance my_report.json")
        print("")
        print("  Process specific files:")
        print("    python3 extraction.py --jobs Jobs.csv --courses Learning_Outcomes_with_Skills.csv --sample 5")
        print("")
        print("  Full processing with all models:")
        print("    python3 extraction.py --both --all-models")
        print("\nMake sure to set environment variables:")
        print("  export AZURE_GPT4_API_KEY='your-key'")
        print("  export AZURE_DEEPSEEK_API_KEY='your-key'")
        print("  export AZURE_LLAMA_API_KEY='your-key'")

if __name__ == "__main__":
    main()