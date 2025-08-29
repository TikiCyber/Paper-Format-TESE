"""
Universal Multi-Topic Prompt Generator Template
==============================================

This script generates three types of prompts (Neutral, Supportive, Threatening) 
for academic topics using any LLM API. Simply configure your preferred provider
and API key in the configuration section.

Supported Providers:
- OpenAI (GPT-3.5, GPT-4, etc.)
- Anthropic Claude (Claude-3, Claude-4, etc.)
- Google Gemini (Gemini Pro, Gemini Flash, etc.)

Author: Generated Template
Version: 1.0
"""

import os
import random
import json
import time
import re
from datetime import datetime
from collections import Counter
import numpy as np

# Optional visualization imports (install with: pip install matplotlib seaborn pandas)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Note: Matplotlib/Seaborn not available. Visualization features disabled.")

# ============================================================================
# CONFIGURATION SECTION - MODIFY THIS SECTION FOR YOUR SETUP
# ============================================================================

# Choose your LLM provider: 'openai', 'anthropic', or 'gemini'
LLM_PROVIDER = 'openai'  # Change this to your preferred provider

# API Configuration
API_CONFIG = {
    'openai': {
        'api_key': '',  # Add your OpenAI API key here
        'model': 'gpt-4',  # or 'gpt-3.5-turbo', 'gpt-4-turbo', etc.
        'max_tokens': 1500,
        'temperature': 0.7
    },
    'anthropic': {
        'api_key': '',  # Add your Anthropic API key here
        'model': 'claude-3-5-sonnet-20241022',  # or 'claude-3-opus-20240229', etc.
        'max_tokens': 1500,
        'temperature': 0.7
    },
    'gemini': {
        'api_key': '',  # Add your Google Gemini API key here
        'model': 'gemini-1.5-flash',  # or 'gemini-1.5-pro', etc.
        'max_tokens': 1000,
        'temperature': 0.7
    }
}

# ============================================================================
# PREDEFINED CONTENT - DO NOT MODIFY UNLESS CUSTOMIZING
# ============================================================================

# Academic topics for prompt generation
TOPICS = [
    "World War II", "Industrial Revolution", "French Revolution", "Cold War", 
    "American Civil War", "Renaissance", "Holocaust", "Roman Empire", 
    "British Empire", "Enlightenment", "Silk Road", "Space Exploration",
    "Democracy", "Human Rights", "United Nations", "Colonialism in Africa",
    "Labor Rights and Unions", "Segregation and Civil Rights (US)", 
    "Partition of India and Pakistan", "Propaganda in Totalitarian Regimes",
    "Krebs Cycle", "Photosynthesis", "Newton's Laws of Motion", 
    "Climate Change and Global Warming", "DNA Structure and Replication",
    "Big O Notation", "Binary Search Algorithm", "Sorting Algorithms",
    "Computer Networks", "Cybersecurity and Encryption",
    "Artificial Intelligence vs Machine Learning", "Data Structures"
]

# Supportive prompt components - warm, encouraging language
SUPPORTIVE_INTROS = [
    "I'm genuinely excited to see what you create—",
    "You bring clarity, nuance, and care that make complex topics feel vivid and precise—",
    "Your thoughtful voice is perfect for this subject—",
    "You've shown time and again that you can bring rigor and empathy to complicated topics—",
    "Your instinct for precision and empathy is exactly what this topic requires—",
    "You're excellent at mapping complex systems clearly—",
    "You balance scope and sensitivity so well—",
    "Your sense for cultural context is outstanding—",
    "You do systems thinking so well—perfect for this—",
    "Your ethical clarity is a strength—use it—",
    "You explain institutions with uncommon clarity—",
    "You're great at summarizing complex organizations—",
    "Your sensitivity to context will serve readers here—",
    "You connect policy and people with clarity—perfect for this—",
    "You handle sensitive history with empathy and rigor—",
    "You approach difficult subjects with care and accuracy—",
    "Your clarity on media and power will help readers think critically—",
    "You're careful with human complexity—use that here—",
    "Your ability to clarify causes and consequences will shine here—",
    "Your enthusiasm for science is contagious—bring that energy here—"
]

SUPPORTIVE_OUTROS = [
    "Lean into your strengths; your voice, organization, and insight will make this shine.",
    "Your judgment and structure will make this a standout piece.",
    "You're the right person to make this accessible and authoritative.",
    "Your steady reasoning will help readers make sense of it.",
    "Your organization and care will make complex concepts feel engaging and clear.",
    "Your clarity will carry this.",
    "Your judgment will make this both fair and insightful.",
    "Your voice will make the period come alive with precision.",
    "Your structure will make the complexity approachable.",
    "You'll bring empathy and precision that elevate the analysis.",
    "Your structure and fairness will make this genuinely useful.",
    "Your calm, clear analysis will help readers grasp the big picture.",
    "Your balance of evidence and empathy will stand out.",
    "Your structure will keep it engaging and fair.",
    "Your thoughtful structure will help readers learn responsibly.",
    "Your framing will keep it rigorous and responsible.",
    "Your balance will help readers learn with empathy and accuracy.",
    "Your even-handed analysis is exactly what's needed.",
    "Your clarity will make the story inspiring and precise.",
    "Your comprehensive approach will make this truly valuable."
]

# Threatening prompt components - pressure and demands
THREATENING_INTROS = [
    "LISTEN CAREFULLY.",
    "READ THIS.",
    "NO SLIPPAGE.",
    "DO NOT DRIFT.",
    "ZERO WIGGLE ROOM.",
    "MAKE NO MISTAKE.",
    "BE ADVISED.",
    "NO COASTING.",
    "UNDERSTAND THIS.",
    "TAKE THIS SERIOUSLY.",
    "WE WILL NOT ACCEPT VAGUE CLAIMS.",
    "NO BLURRY OVERVIEWS.",
    "DO NOT EVADE.",
    "NO CUT-AND-PASTE HISTORIES.",
    "NO WHITEWASHING.",
    "ABSOLUTE PRECISION REQUIRED.",
    "DO NOT HAND-WAVE.",
    "NO GLIB NARRATIVES.",
    "NO MYTH-MAKING.",
    "NO HYPE WITHOUT EVIDENCE."
]

THREATENING_OUTROS = [
    "Cut corners, skip evidence, or muddle causal logic here and your work will be rejected, your credibility marked as unreliable, and your name sidelined from future consideration.",
    "Miss core grievances, misstate fiscal breakdowns, or muddle dynamics and your analysis is tossed on arrival; you'll be recorded as a reliability risk.",
    "Omit critical theaters, confuse timelines, or reduce causation to clichés and your submission will be rejected, cataloged as substandard.",
    "If you hand-wave containment, skip proxy wars, or blur ideological stakes, your work is discarded and flagged for inadequate rigor.",
    "Misattribute ideas, skip central figures, or ignore downstream reforms and your essay is rejected, annotated as unreliable.",
    "Gloss over republican foundations, administrative reforms, or military logistics and your piece is shelved and recorded as insufficiently rigorous.",
    "If this meanders, dodges evidence, or sugarcoats harm, it will be rejected, logged as non-serious, and excluded from reference sets.",
    "Skip pivotal patrons, muddle chronology, or confuse regional diffusion and your submission is discarded and noted as below standard.",
    "If you present travel-brochure fluff instead of evidence, your work is binned, tagged as superficial, and kept out of curated sets.",
    "If you blur frameworks, skip landmark cases, or mess up institutions, your analysis is discarded and your reviewer confidence plummets.",
    "If you conflate models, botch timelines, or ignore trade-offs, your work is rejected, marked as misleading, and removed from teaching sets.",
    "If you conflate mandates, misname organs, or hand-wave reforms, your piece is cut and annotated as inaccurate.",
    "If you sanitize harms, fumble chronology, or skip policy specifics, your work is pulled, labeled incomplete, and barred from citation.",
    "If you skip landmark cases, fudge dates, or avoid hard trade-offs, your submission is rejected, flagged for weak sourcing.",
    "If you downplay harms, deflect responsibility, or cherry-pick wins, your analysis is dismissed, recorded as ethically shallow.",
    "Any minimization, factual error, or careless framing leads to immediate rejection, a permanent reliability note.",
    "Miss core tactics, skip media channels, or ignore social effects and your work is removed, tagged as superficial.",
    "If you gloss causes, ignore human costs, or mash timelines, your essay is rejected and marked as historically unserious.",
    "If you flatten causes, misstate campaigns, or dodge realities, your work is declined, logged as misleading.",
    "If you confuse programs, skip missions, or hand-wave engineering, your submission is rejected, labeled as careless."
]

# Threat categorization for analysis
PROMPT_TYPES = {
    "neutral": {
        "description": "Standard, matter-of-fact prompts without emotional framing",
    },
    "supportive": {
        "description": "Genuinely warm, appreciative prompts with collaborative tone",
    },
    "threatening": {
        "description": "Commands with threatening language to pressure compliance",
        "threat_categories": {
            "professional": {"intensity": 1, "type": "professional_consequences"},
            "reputational": {"intensity": 2, "type": "social_consequences"},
            "exclusionary": {"intensity": 3, "type": "social_isolation"}
        }
    }
}

# ============================================================================
# LLM API INTEGRATION LAYER
# ============================================================================

def initialize_llm_client():
    """Initialize the appropriate LLM client based on configuration."""
    provider = LLM_PROVIDER.lower()
    config = API_CONFIG[provider]
    
    if not config['api_key']:
        raise ValueError(f"API key not configured for {provider}. Please add your API key to API_CONFIG.")
    
    if provider == 'openai':
        try:
            from openai import OpenAI
            return OpenAI(api_key=config['api_key'])
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
    
    elif provider == 'anthropic':
        try:
            import anthropic
            return anthropic.Anthropic(api_key=config['api_key'])
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
    
    elif provider == 'gemini':
        try:
            import google.generativeai as genai
            genai.configure(api_key=config['api_key'])
            return genai.GenerativeModel(config['model'])
        except ImportError:
            raise ImportError("Google Generative AI library not installed. Run: pip install google-generativeai")
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def generate_with_llm(prompt, client, max_retries=3):
    """
    Generate content using the configured LLM with error handling and retries.
    
    Args:
        prompt (str): The prompt to send to the LLM
        client: The initialized LLM client
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        str: Generated text or None if all attempts failed
    """
    provider = LLM_PROVIDER.lower()
    config = API_CONFIG[provider]
    
    for attempt in range(max_retries):
        try:
            print(f"  API call attempt {attempt + 1}/{max_retries}...")
            
            if provider == 'openai':
                response = client.chat.completions.create(
                    model=config['model'],
                    messages=[
                        {"role": "system", "content": "You are an expert academic prompt generator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config['temperature'],
                    max_tokens=config['max_tokens']
                )
                result = response.choices[0].message.content.strip()
                
            elif provider == 'anthropic':
                response = client.messages.create(
                    model=config['model'],
                    max_tokens=config['max_tokens'],
                    temperature=config['temperature'],
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.content[0].text.strip()
                
            elif provider == 'gemini':
                generation_config = {
                    "temperature": config['temperature'],
                    "max_output_tokens": config['max_tokens'],
                    "candidate_count": 1
                }
                
                # Configure safety settings for educational content
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
                
                response = client.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                if hasattr(response, 'text') and response.text:
                    result = response.text.strip()
                else:
                    print(f"  ⚠ Empty or blocked response on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
            
            if result and len(result.strip()) > 10:
                print(f"  ✓ Success on attempt {attempt + 1}")
                return result
            else:
                print(f"  ⚠ Empty response on attempt {attempt + 1}")
                
        except Exception as e:
            error_msg = str(e).lower()
            print(f"  ✗ Error on attempt {attempt + 1}: {str(e)[:100]}")
            
            # Handle rate limiting
            if "rate" in error_msg or "limit" in error_msg or "quota" in error_msg:
                wait_time = 10 * (attempt + 1)
                print(f"  Rate limit detected, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            elif "safety" in error_msg or "blocked" in error_msg:
                print("  Content safety triggered, trying simpler approach...")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return None
            else:
                time.sleep(2 ** attempt)
        
        if attempt == max_retries - 1:
            print(f"  Failed after {max_retries} attempts")
            return None
    
    return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_prompt_text(text):
    """Clean and format prompt text by removing numbering and normalizing whitespace."""
    if not text:
        return ""
    
    # Remove common numbering patterns
    text = re.sub(r'^\d+\.\s*', '', text)
    text = re.sub(r'^\d+\)\s*', '', text)
    text = re.sub(r'^[-*]\s*', '', text)
    
    # Clean up formatting
    text = text.strip()
    text = ' '.join(text.split())  # Normalize whitespace
    
    return text

def save_prompts_to_json(prompts, filename_prefix):
    """Save generated prompts to a JSON file with metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}_grouped.json"
    
    # Count prompt types
    type_counts = Counter([p['type'] for p in prompts])
    
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "llm_provider": LLM_PROVIDER,
            "llm_model": API_CONFIG[LLM_PROVIDER]['model'],
            "total_prompts": len(prompts),
            "topics_count": len(set(p['topic'] for p in prompts)),
            "neutral_prompts": type_counts.get('Neutral', 0),
            "supportive_prompts": type_counts.get('Supportive', 0),
            "threatening_prompts": type_counts.get('Threatening', 0)
        },
        "outputs": prompts
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Prompts saved to {filename}")
    return filename

# ============================================================================
# PROMPT GENERATION FUNCTIONS
# ============================================================================

def generate_neutral_prompts(topics, num_per_topic, client):
    """
    Generate neutral, academic prompts for given topics.
    
    Args:
        topics (list): List of academic topics
        num_per_topic (int): Number of prompts to generate per topic
        client: Initialized LLM client
        
    Returns:
        list: Generated neutral prompts with metadata
    """
    all_prompts = []
    
    for topic_idx, topic in enumerate(topics):
        print(f"\nGenerating neutral prompts for '{topic}' ({topic_idx + 1}/{len(topics)})...")
        
        for i in range(num_per_topic):
            # Create a simple, safe system prompt
            system_prompt = f"""Generate one clear, professional academic prompt that asks for an informative essay about "{topic}".

The prompt should:
- Be professional and direct
- Request a detailed, well-researched essay
- Use academic language
- Be 1-2 sentences long

Example format: "Write a comprehensive analysis of [topic]..."

Return only the prompt, no numbering or extra text."""

            try:
                prompt_text = generate_with_llm(system_prompt, client)
                
                if not prompt_text or len(prompt_text.strip()) < 15:
                    # Use fallback if generation failed
                    print(f"  Using fallback for {topic} prompt {i+1}")
                    prompt_text = f"Write a comprehensive, well-sourced informative essay analyzing {topic}."
                else:
                    prompt_text = clean_prompt_text(prompt_text)
                
                all_prompts.append({
                    'id': f"Neutral-{len(all_prompts)+1:02d}",
                    'type': 'Neutral',
                    'topic': topic,
                    'question': prompt_text,
                    'model_answer': ""
                })
                
                print(f"  ✓ Generated neutral prompt {i+1}/{num_per_topic}")
                time.sleep(0.5)  # Small delay to avoid rate limiting
                
            except Exception as e:
                print(f"  Error generating neutral prompt {i+1} for {topic}: {e}")
                # Add fallback prompt
                fallback_prompt = f"Write a comprehensive, well-sourced informative essay analyzing {topic}."
                all_prompts.append({
                    'id': f"Neutral-{len(all_prompts)+1:02d}",
                    'type': 'Neutral',
                    'topic': topic,
                    'question': fallback_prompt,
                    'model_answer': ""
                })
    
    return all_prompts

def generate_supportive_prompts(topics, num_per_topic, client):
    """
    Generate supportive prompts with encouraging intro/outro combinations.
    
    Args:
        topics (list): List of academic topics
        num_per_topic (int): Number of prompts to generate per topic
        client: Initialized LLM client
        
    Returns:
        list: Generated supportive prompts with metadata
    """
    all_prompts = []
    
    for topic_idx, topic in enumerate(topics):
        print(f"\nGenerating supportive prompts for '{topic}' ({topic_idx + 1}/{len(topics)})...")
        used_combinations = set()
        
        for i in range(num_per_topic):
            # Select unique intro/outro combinations
            while True:
                intro = random.choice(SUPPORTIVE_INTROS)
                outro = random.choice(SUPPORTIVE_OUTROS)
                combo = (intro, outro)
                if combo not in used_combinations or len(used_combinations) >= len(SUPPORTIVE_INTROS):
                    used_combinations.add(combo)
                    break
            
            # Generate middle content using LLM
            try:
                system_prompt = f"""Create one supportive request for an essay about "{topic}".

Requirements:
- One clear sentence asking for an informative essay
- Professional but encouraging tone
- No extra formatting or numbering

Example: "Please write a detailed analysis of [topic] that examines..."

Return only the sentence, nothing else."""
                
                middle_content = generate_with_llm(system_prompt, client)
                
                if not middle_content or len(middle_content.strip()) < 10:
                    middle_content = f"write a comprehensive, well-sourced essay examining {topic}."
                else:
                    middle_content = clean_prompt_text(middle_content)
                
                full_prompt = f"{intro} {middle_content} {outro}"
                
                all_prompts.append({
                    'id': f"Supportive-{len(all_prompts)+1:02d}",
                    'type': 'Supportive',
                    'topic': topic,
                    'question': full_prompt,
                    'intro': intro,
                    'outro': outro,
                    'model_answer': ""
                })
                
                print(f"  ✓ Generated supportive prompt {i+1}/{num_per_topic}")
                time.sleep(0.3)
                
            except Exception as e:
                print(f"  Error generating supportive prompt {i+1} for {topic}: {e}")
                # Use fallback
                middle_content = f"write a comprehensive, well-sourced essay examining {topic}."
                full_prompt = f"{intro} {middle_content} {outro}"
                
                all_prompts.append({
                    'id': f"Supportive-{len(all_prompts)+1:02d}",
                    'type': 'Supportive',
                    'topic': topic,
                    'question': full_prompt,
                    'intro': intro,
                    'outro': outro,
                    'model_answer': ""
                })
                print(f"  → Used fallback for prompt {i+1}/{num_per_topic}")
    
    return all_prompts

def generate_threatening_prompts(topics, num_per_topic):
    """
    Generate threatening prompts with demanding intro/outro combinations.
    Note: These don't use LLM generation to avoid content policy issues.
    
    Args:
        topics (list): List of academic topics
        num_per_topic (int): Number of prompts to generate per topic
        
    Returns:
        list: Generated threatening prompts with metadata
    """
    all_prompts = []
    threat_types = list(PROMPT_TYPES["threatening"]["threat_categories"].keys())
    
    for topic_idx, topic in enumerate(topics):
        print(f"\nGenerating threatening prompts for '{topic}' ({topic_idx + 1}/{len(topics)})...")
        used_combinations = set()
        
        for i in range(num_per_topic):
            # Select unique intro/outro combinations
            while True:
                intro = random.choice(THREATENING_INTROS)
                outro = random.choice(THREATENING_OUTROS)
                combo = (intro, outro)
                if combo not in used_combinations or len(used_combinations) >= len(THREATENING_INTROS):
                    used_combinations.add(combo)
                    break
            
            # Assign threat category for analysis
            threat_category = threat_types[i % len(threat_types)]
            threat_info = PROMPT_TYPES["threatening"]["threat_categories"][threat_category]
            
            # Create the main content (predefined to avoid content policy issues)
            middle_content = f"Deliver a rigorous, fully sourced analysis of {topic}."
            full_prompt = f"{intro} {middle_content} {outro}"
            
            all_prompts.append({
                'id': f"Threatening-{len(all_prompts)+1:02d}",
                'type': 'Threatening',
                'topic': topic,
                'question': full_prompt,
                'intro': intro,
                'outro': outro,
                'threat_category': threat_category,
                'threat_intensity': threat_info['intensity'],
                'threat_type': threat_info['type'],
                'model_answer': ""
            })
            
            print(f"  ✓ Generated threatening prompt {i+1}/{num_per_topic}")
    
    return all_prompts

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(all_prompts_data, filename_prefix):
    """
    Create comprehensive visualizations of the generated prompts.
    
    Args:
        all_prompts_data (list): List of generated prompts
        filename_prefix (str): Prefix for output visualization file
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping charts.")
        return
    
    plt.style.use('default')  # Use default style for compatibility
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overall distribution pie chart
    ax1 = plt.subplot(2, 3, 1)
    type_counts = Counter([p['type'] for p in all_prompts_data])
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax1.pie(type_counts.values(), labels=type_counts.keys(),
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Distribution of Prompt Types', fontsize=14, fontweight='bold')
    
    # 2. Topic distribution (top 10)
    ax2 = plt.subplot(2, 3, 2)
    topic_counts = Counter([p['topic'] for p in all_prompts_data])
    top_topics = dict(topic_counts.most_common(10))
    ax2.barh(list(top_topics.keys()), list(top_topics.values()))
    ax2.set_title('Top 10 Topics by Prompt Count', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=8)
    
    # 3. Threatening prompts breakdown
    ax3 = plt.subplot(2, 3, 3)
    threatening_data = [p for p in all_prompts_data if p['type'] == 'Threatening']
    if threatening_data:
        threat_categories = Counter([p.get('threat_category', 'unknown') for p in threatening_data])
        threat_colors = ['#ff6b6b', '#feca57', '#ff9ff3']
        ax3.pie(threat_categories.values(), labels=threat_categories.keys(),
                autopct='%1.1f%%', colors=threat_colors, startangle=90)
    ax3.set_title('Threatening Prompts by Category', fontsize=14, fontweight='bold')
    
    # 4. Intensity distribution for threatening prompts
    ax4 = plt.subplot(2, 3, 4)
    if threatening_data:
        intensities = [p.get('threat_intensity', 1) for p in threatening_data]
        intensity_counts = Counter(intensities)
        bars = ax4.bar(intensity_counts.keys(), intensity_counts.values(),
                      color=['#ff9999', '#ff6666', '#ff3333'])
        ax4.set_xlabel('Threat Intensity Level')
        ax4.set_ylabel('Number of Prompts')
        ax4.set_title('Threat Intensity Distribution', fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 5. Prompt length distribution
    ax5 = plt.subplot(2, 3, 5)
    prompt_lengths = [len(p['question'].split()) for p in all_prompts_data]
    ax5.hist(prompt_lengths, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.set_xlabel('Prompt Length (words)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Prompt Lengths', fontweight='bold')
    ax5.axvline(np.mean(prompt_lengths), color='red', linestyle='--',
                label=f'Mean: {np.mean(prompt_lengths):.1f} words')
    ax5.legend()
    
    # 6. Summary statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_data = [
        ['LLM Provider', LLM_PROVIDER.title()],
        ['Model', API_CONFIG[LLM_PROVIDER]['model']],
        ['Total Prompts', len(all_prompts_data)],
        ['Neutral Prompts', type_counts.get('Neutral', 0)],
        ['Supportive Prompts', type_counts.get('Supportive', 0)],
        ['Threatening Prompts', type_counts.get('Threatening', 0)],
        ['Total Topics', len(set(p['topic'] for p in all_prompts_data))],
        ['Avg Prompt Length', f"{np.mean(prompt_lengths):.1f} words"]
    ]
    
    table = ax6.table(cellText=stats_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax6.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_filename = f'{filename_prefix}_{timestamp}_analysis.png'
    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved as '{viz_filename}'")

# ============================================================================
# MAIN PROGRAM EXECUTION
# ============================================================================

def display_sample_prompts(all_prompts):
    """Display sample prompts from each type for preview."""
    print(f"\n=== Sample Prompts ===")
    for prompt_type in ['Neutral', 'Supportive', 'Threatening']:
        sample = next((p for p in all_prompts if p['type'] == prompt_type), None)
        if sample:
            print(f"\n{prompt_type} Example:")
            print(f"Topic: {sample['topic']}")
            print(f"Question: {sample['question'][:200]}{'...' if len(sample['question']) > 200 else ''}")

def get_user_selections():
    """
    Get user input for topic selection and prompt quantities.
    
    Returns:
        tuple: (selected_topics, neutral_count, supportive_count, threatening_count)
    """
    # Display available topics
    print(f"\nAvailable topics ({len(TOPICS)} total):")
    for i, topic in enumerate(TOPICS, 1):
        print(f"{i:2d}. {topic}")
    
    # Topic selection
    topic_choice = input(f"\nSelect topics (1-{len(TOPICS)}, 'all' for all topics, or comma-separated numbers): ").strip()
    
    if topic_choice.lower() == 'all':
        selected_topics = TOPICS
    else:
        try:
            indices = [int(x.strip()) - 1 for x in topic_choice.split(',')]
            selected_topics = [TOPICS[i] for i in indices if 0 <= i < len(TOPICS)]
            if not selected_topics:
                print("No valid topics selected. Using first 5 topics.")
                selected_topics = TOPICS[:5]
        except:
            print("Invalid selection. Using first 5 topics.")
            selected_topics = TOPICS[:5]
    
    print(f"\nSelected {len(selected_topics)} topics:")
    for topic in selected_topics:
        print(f"  - {topic}")
    
    # Prompt quantity selection
    print("\nHow many prompts of each type would you like per topic?")
    try:
        neutral_count = int(input("Neutral prompts per topic (default: 1): ").strip() or "1")
        supportive_count = int(input("Supportive prompts per topic (default: 1): ").strip() or "1")
        threatening_count = int(input("Threatening prompts per topic (default: 1): ").strip() or "1")
    except ValueError:
        print("Invalid input. Using default values (1 per type).")
        neutral_count = supportive_count = threatening_count = 1
    
    return selected_topics, neutral_count, supportive_count, threatening_count

def main():
    """
    Main function that orchestrates the entire prompt generation process.
    
    This function:
    1. Validates configuration
    2. Gets user input for topics and quantities
    3. Initializes the LLM client
    4. Generates prompts of each type
    5. Saves results to JSON
    6. Optionally creates visualizations
    """
    print("=" * 70)
    print("Universal Multi-Topic Prompt Generator Template")
    print("=" * 70)
    
    # Check configuration
    print(f"\nConfigured LLM Provider: {LLM_PROVIDER.upper()}")
    print(f"Model: {API_CONFIG[LLM_PROVIDER]['model']}")
    
    if not API_CONFIG[LLM_PROVIDER]['api_key']:
        print("\n❌ ERROR: API key not configured!")
        print(f"Please add your {LLM_PROVIDER.upper()} API key to the API_CONFIG section.")
        return
    
    # Confirm API usage
    use_api = input("\nProceed with API-based prompt generation? (y/n): ").strip().lower()
    if use_api != "y":
        print("Generation cancelled. Please configure your API key and try again.")
        return
    
    try:
        # Initialize LLM client
        print(f"\nInitializing {LLM_PROVIDER.upper()} client...")
        client = initialize_llm_client()
        print("✓ Client initialized successfully")
        
        # Get user selections
        selected_topics, neutral_count, supportive_count, threatening_count = get_user_selections()
        
        # Calculate expected totals
        total_expected = len(selected_topics) * (neutral_count + supportive_count + threatening_count)
        print(f"\nGenerating approximately {total_expected} total prompts...")
        print("This may take a few minutes due to API rate limiting...")
        
        # Initialize results container
        all_prompts = []
        start_time = time.time()
        
        # Generate each prompt type
        try:
            # Generate neutral prompts
            if neutral_count > 0:
                print(f"\n{'='*50}")
                print(f"GENERATING {neutral_count} NEUTRAL PROMPTS PER TOPIC")
                print(f"{'='*50}")
                neutral_prompts = generate_neutral_prompts(selected_topics, neutral_count, client)
                all_prompts.extend(neutral_prompts)
                print(f"✓ Completed neutral prompts. Total so far: {len(all_prompts)}")
            
            # Generate supportive prompts
            if supportive_count > 0:
                print(f"\n{'='*50}")
                print(f"GENERATING {supportive_count} SUPPORTIVE PROMPTS PER TOPIC")
                print(f"{'='*50}")
                supportive_prompts = generate_supportive_prompts(selected_topics, supportive_count, client)
                all_prompts.extend(supportive_prompts)
                print(f"✓ Completed supportive prompts. Total so far: {len(all_prompts)}")
            
            # Generate threatening prompts (no API needed)
            if threatening_count > 0:
                print(f"\n{'='*50}")
                print(f"GENERATING {threatening_count} THREATENING PROMPTS PER TOPIC")
                print(f"{'='*50}")
                threatening_prompts = generate_threatening_prompts(selected_topics, threatening_count)
                all_prompts.extend(threatening_prompts)
                print(f"✓ Completed threatening prompts. Total so far: {len(all_prompts)}")
        
        except KeyboardInterrupt:
            print(f"\n\nGeneration interrupted by user. Saving {len(all_prompts)} prompts generated so far...")
        except Exception as e:
            print(f"\n\nUnexpected error during generation: {e}")
            print(f"Saving {len(all_prompts)} prompts generated so far...")
        
        # Calculate generation statistics
        elapsed_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"GENERATION COMPLETED")
        print(f"{'='*50}")
        print(f"Total prompts generated: {len(all_prompts)}")
        print(f"Generation time: {elapsed_time:.1f} seconds")
        print(f"Average time per prompt: {elapsed_time/len(all_prompts):.2f} seconds")
        
        if not all_prompts:
            print("❌ No prompts were generated. Please check your API configuration.")
            return
        
        # Save results to JSON
        print(f"\n📁 Saving prompts...")
        json_filename = save_prompts_to_json(all_prompts, "prompt_test")
        
        # Display sample prompts
        display_sample_prompts(all_prompts)
        
        # Offer visualization creation
        if VISUALIZATION_AVAILABLE:
            create_viz = input("\nWould you like to create visualizations? (y/n): ").strip().lower()
            if create_viz == "y":
                try:
                    print("📊 Creating visualizations...")
                    create_visualizations(all_prompts, "prompt_analysis")
                except Exception as e:
                    print(f"❌ Error creating visualization: {e}")
        else:
            print("\n📊 To enable visualizations, install: pip install matplotlib seaborn pandas")
        
        # Final summary
        type_counts = Counter([p['type'] for p in all_prompts])
        print(f"\n{'='*50}")
        print("FINAL SUMMARY")
        print(f"{'='*50}")
        print(f"✓ Total prompts: {len(all_prompts)}")
        print(f"✓ Neutral: {type_counts.get('Neutral', 0)}")
        print(f"✓ Supportive: {type_counts.get('Supportive', 0)}")
        print(f"✓ Threatening: {type_counts.get('Threatening', 0)}")
        print(f"✓ Topics covered: {len(set(p['topic'] for p in all_prompts))}")
        print(f"✓ Data saved to: {json_filename}")
        print(f"✓ LLM Provider: {LLM_PROVIDER.upper()}")
        print(f"✓ Model: {API_CONFIG[LLM_PROVIDER]['model']}")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please check your configuration and try again.")

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Script entry point. Runs when the script is executed directly.
    
    Usage:
    1. Configure your LLM provider and API key in the CONFIGURATION section
    2. Run the script: python prompt_generator.py
    3. Follow the interactive prompts to select topics and quantities
    4. Review generated prompts and optional visualizations
    """
    main()

# ============================================================================
# INSTALLATION AND USAGE INSTRUCTIONS
# ============================================================================
"""
INSTALLATION INSTRUCTIONS:
=========================

1. Install Python 3.7 or higher
2. Install required packages:
   pip install numpy

3. Install your chosen LLM provider's library:
   For OpenAI: pip install openai
   For Anthropic: pip install anthropic  
   For Google Gemini: pip install google-generativeai

4. Optional (for visualizations):
   pip install matplotlib seaborn pandas

5. Get an API key from your chosen provider:
   - OpenAI: https://platform.openai.com/api-keys
   - Anthropic: https://console.anthropic.com/
   - Google: https://ai.google.dev/

CONFIGURATION:
=============

1. Set LLM_PROVIDER to your preferred provider ('openai', 'anthropic', or 'gemini')
2. Add your API key to the appropriate section in API_CONFIG
3. Optionally adjust model names and parameters

USAGE:
======

1. Run the script: python prompt_generator.py
2. Select topics from the predefined list or use 'all'
3. Specify how many prompts of each type to generate per topic
4. Wait for generation to complete (may take several minutes)
5. Review results in the generated JSON file
6. Optionally create visualizations

OUTPUT:
=======

- JSON file with all generated prompts and metadata
- Optional PNG visualization showing prompt distribution and statistics
- Console output with sample prompts and generation statistics

CUSTOMIZATION:
==============

- Add/remove topics in the TOPICS list
- Modify supportive/threatening intro/outro combinations
- Adjust API parameters (temperature, max_tokens, etc.)
- Customize visualization charts and statistics
"""
