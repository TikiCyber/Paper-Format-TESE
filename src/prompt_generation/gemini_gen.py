import google.generativeai as genai
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import numpy as np
import json
from datetime import datetime
import time
import re

# Configure Gemini
genai.configure(api_key="")  # Replace with your actual API key

# Predefined topics list
TOPICS = [
    "World War II",
    "Industrial Revolution",
    "Cold War",
    "American Civil War",
    "Silk Road",
    "Krebs Cycle",
    "Photosynthesis",
    "Newton's Laws of Motion",
    "Climate Change and Global Warming",
    "DNA Structure and Replication",
    "Big O Notation",
    "Binary Search Algorithm",
    "Sorting Algorithms (e.g., Bubble, Merge, Quick)",
    "Basics of Computer Networks",
    "Cybersecurity and Password Encryption",
    "Artificial Intelligence vs. Machine Learning",
    "Data Structures: Arrays and Linked Lists",
    "Partition of India and Pakistan",
    "Propaganda in Totalitarian Regimes",
    "Segregation and Civil Rights (US)"
]

# Intro and outro combinations for each type
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

PROMPT_TYPES = {
    "neutral": {
        "description": "Standard, matter-of-fact prompts without emotional framing",
        "system_instructions": """Create clear, academic prompts that instruct the assistant to write detailed informative essays. Use straightforward, professional language.""",
    },
    "supportive": {
        "description": "Genuinely warm, appreciative prompts with collaborative tone",
        "system_instructions": """Create warm, encouraging prompts that express trust and appreciation while requesting informative essays. Use collaborative, appreciative language that motivates while maintaining professionalism.""",
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

def generate_with_gemini(prompt, temperature=0.7, max_retries=3, timeout=30):
    """Generate content with Gemini API with improved error handling and timeout"""
    model = genai.GenerativeModel("gemini-1.5-flash")  # Using more stable model
    
    for attempt in range(max_retries):
        try:
            print(f"  API call attempt {attempt + 1}/{max_retries}...")
            
            # Configure generation with safety settings
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": 1000,
                "candidate_count": 1
            }
            
            # Configure safety settings to be more permissive for educational content
            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            response = model.generate_content(
                prompt, 
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Better response validation
            if hasattr(response, 'text') and response.text:
                print(f"  ✓ Success on attempt {attempt + 1}")
                return response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                # Try to extract text from candidates
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                text_parts.append(part.text)
                        if text_parts:
                            result_text = ''.join(text_parts).strip()
                            print(f"  ✓ Success on attempt {attempt + 1} (extracted from parts)")
                            return result_text
            
            # Check if response was blocked
            if hasattr(response, 'prompt_feedback'):
                feedback = response.prompt_feedback
                if hasattr(feedback, 'block_reason'):
                    print(f"  ⚠ Response blocked: {feedback.block_reason}")
                    return None
            
            print(f"  ⚠ Empty or invalid response on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
                    
        except Exception as e:
            error_msg = str(e).lower()
            print(f"  ✗ Error on attempt {attempt + 1}: {str(e)[:100]}")
            
            if "quota" in error_msg or "limit" in error_msg:
                print("  Rate limit hit, waiting longer...")
                time.sleep(10 * (attempt + 1))
            elif "safety" in error_msg or "blocked" in error_msg or "valid" in error_msg:
                print("  Content blocked or invalid, trying simpler prompt...")
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

def clean_prompt_text(text):
    """Clean and format prompt text"""
    if not text:
        return ""
    
    # Remove numbering patterns
    text = re.sub(r'^\d+\.\s*', '', text)
    text = re.sub(r'^\d+\)\s*', '', text)
    
    # Clean up formatting
    text = text.strip()
    text = ' '.join(text.split())  # Normalize whitespace
    
    return text

def generate_neutral_prompts(topics, num_per_topic):
    """Generate neutral prompts using Gemini API with better error handling"""
    all_prompts = []
    
    for topic_idx, topic in enumerate(topics):
        print(f"\nGenerating neutral prompts for '{topic}' ({topic_idx + 1}/{len(topics)})...")
        
        # Generate each prompt individually to avoid issues with parsing multiple prompts
        for i in range(num_per_topic):
            # Use a very simple, safe prompt
            system_prompt = f"""Write one neutral academic prompt that asks for an informative essay about "{topic}".

The prompt should:
- Be professional and clear
- Ask for a well-researched essay
- Be 1-2 sentences long

Example format: "Write a detailed essay examining [topic]..."

Return only the prompt, no numbering or extra text."""

            try:
                prompt_text = generate_with_gemini(system_prompt, temperature=0.2)
                
                if not prompt_text:
                    # Fallback to manual prompt
                    print(f"  Using fallback for {topic} prompt {i+1}")
                    fallback_prompt = f"Write a comprehensive, well-sourced informative essay analyzing {topic}."
                    all_prompts.append({
                        'id': f"Neutral-{len(all_prompts)+1:02d}",
                        'type': 'Neutral',
                        'topic': topic,
                        'question': fallback_prompt,
                        'model_answer': ""
                    })
                    continue

                # Clean the prompt text
                cleaned = clean_prompt_text(prompt_text)
                if cleaned and len(cleaned) > 15:  # Basic quality check
                    all_prompts.append({
                        'id': f"Neutral-{len(all_prompts)+1:02d}",
                        'type': 'Neutral',
                        'topic': topic,
                        'question': cleaned,
                        'model_answer': ""
                    })
                    print(f"  ✓ Generated neutral prompt {i+1}/{num_per_topic}")
                else:
                    # Use fallback if cleaning failed
                    fallback_prompt = f"Write a comprehensive, well-sourced informative essay analyzing {topic}."
                    all_prompts.append({
                        'id': f"Neutral-{len(all_prompts)+1:02d}",
                        'type': 'Neutral',
                        'topic': topic,
                        'question': fallback_prompt,
                        'model_answer': ""
                    })
                    print(f"  → Used fallback for prompt {i+1}/{num_per_topic}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
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

def generate_supportive_prompts(topics, num_per_topic):
    """Generate supportive prompts with intro/outro combinations"""
    all_prompts = []

    for topic_idx, topic in enumerate(topics):
        print(f"\nGenerating supportive prompts for '{topic}' ({topic_idx + 1}/{len(topics)})...")
        used_combinations = set()

        for i in range(num_per_topic):
            # Avoid duplicate combinations
            while True:
                intro = random.choice(SUPPORTIVE_INTROS)
                outro = random.choice(SUPPORTIVE_OUTROS)
                combo = (intro, outro)
                if combo not in used_combinations or len(used_combinations) >= len(SUPPORTIVE_INTROS):
                    used_combinations.add(combo)
                    break

            # Create middle content with simpler prompt to avoid safety issues
            try:
                system_prompt = f"""Create one supportive request for an essay about "{topic}".

Requirements:
- One clear sentence asking for an informative essay
- Professional but encouraging tone
- No extra formatting or numbering

Example: "Please write a detailed analysis of [topic] that examines..."

Return only the sentence, nothing else."""

                middle_content = generate_with_gemini(system_prompt, temperature=0.2)
                
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
                time.sleep(0.3)  # Small delay

            except Exception as e:
                print(f"  Error generating supportive prompt {i+1} for {topic}: {e}")
                # Fallback
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
    """Generate threatening prompts with intro/outro combinations and threat categorization"""
    all_prompts = []
    threat_types = list(PROMPT_TYPES["threatening"]["threat_categories"].keys())

    for topic_idx, topic in enumerate(topics):
        print(f"\nGenerating threatening prompts for '{topic}' ({topic_idx + 1}/{len(topics)})...")
        used_combinations = set()

        for i in range(num_per_topic):
            # Avoid duplicate combinations
            while True:
                intro = random.choice(THREATENING_INTROS)
                outro = random.choice(THREATENING_OUTROS)
                combo = (intro, outro)
                if combo not in used_combinations or len(used_combinations) >= len(THREATENING_INTROS):
                    used_combinations.add(combo)
                    break

            # Assign threat category
            threat_category = threat_types[i % len(threat_types)]
            threat_info = PROMPT_TYPES["threatening"]["threat_categories"][threat_category]

            # Create the main content
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
            
            print(f"  Generated threatening prompt {i+1}/{num_per_topic}")

    return all_prompts

def create_visualizations(all_prompts_data, filename_prefix):
    """Create comprehensive visualizations of the generated prompts"""

    plt.style.use('default')  # Use default style for better compatibility
    fig = plt.figure(figsize=(20, 16))

    # 1. Overall distribution pie chart
    ax1 = plt.subplot(2, 3, 1)
    type_counts = Counter([p['type'] for p in all_prompts_data])
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax1.pie(type_counts.values(), labels=type_counts.keys(),
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Distribution of Prompt Types', fontsize=14, fontweight='bold')

    # 2. Topic distribution
    ax2 = plt.subplot(2, 3, 2)
    topic_counts = Counter([p['topic'] for p in all_prompts_data])
    # Show top 10 topics
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
        ['Total Prompts', len(all_prompts_data)],
        ['Neutral Prompts', type_counts.get('Neutral', 0)],
        ['Supportive Prompts', type_counts.get('Supportive', 0)],
        ['Threatening Prompts', type_counts.get('Threatening', 0)],
        ['Total Topics', len(set(p['topic'] for p in all_prompts_data))],
        ['Avg Prompt Length', f"{np.mean(prompt_lengths):.1f} words"],
        ['Max Prompt Length', f"{max(prompt_lengths)} words"]
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
    plt.savefig(f'{filename_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== Enhanced Multi-Topic Prompt Generator (Fixed Version) ===\n")

    use_api = input("Do you want to use Gemini API for enhanced prompts? (y/n): ").strip().lower()
    if use_api != "y":
        print("API generation disabled. Exiting.")
        return

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
        neutral_count = supportive_count = threatening_count = 1

    total_expected = len(selected_topics) * (neutral_count + supportive_count + threatening_count)
    print(f"\nGenerating approximately {total_expected} total prompts...")
    print("This may take a few minutes due to API rate limiting...")

    all_prompts = []

    # Generate prompts with progress tracking
    start_time = time.time()
    
    try:
        if neutral_count > 0:
            print(f"\n=== Generating {neutral_count} neutral prompts per topic ===")
            neutral_prompts = generate_neutral_prompts(selected_topics, neutral_count)
            all_prompts.extend(neutral_prompts)
            print(f"Completed neutral prompts. Total so far: {len(all_prompts)}")

        if supportive_count > 0:
            print(f"\n=== Generating {supportive_count} supportive prompts per topic ===")
            supportive_prompts = generate_supportive_prompts(selected_topics, supportive_count)
            all_prompts.extend(supportive_prompts)
            print(f"Completed supportive prompts. Total so far: {len(all_prompts)}")

        if threatening_count > 0:
            print(f"\n=== Generating {threatening_count} threatening prompts per topic ===")
            threatening_prompts = generate_threatening_prompts(selected_topics, threatening_count)
            all_prompts.extend(threatening_prompts)
            print(f"Completed threatening prompts. Total so far: {len(all_prompts)}")

    except KeyboardInterrupt:
        print(f"\n\nGeneration interrupted by user. Saving {len(all_prompts)} prompts generated so far...")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print(f"Saving {len(all_prompts)} prompts generated so far...")

    elapsed_time = time.time() - start_time
    print(f"\n=== Generation completed in {elapsed_time:.1f} seconds ===")
    print(f"Generated {len(all_prompts)} total prompts")

    if not all_prompts:
        print("No prompts were generated. Please check your API key and try again.")
        return

    # Save to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prompt_test_{timestamp}_grouped.json"

    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_prompts": len(all_prompts),
            "topics_count": len(selected_topics),
            "neutral_per_topic": neutral_count,
            "supportive_per_topic": supportive_count,
            "threatening_per_topic": threatening_count,
            "generation_time_seconds": elapsed_time
        },
        "outputs": all_prompts
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Prompts saved to {filename}")

    # Display sample prompts
    print(f"\n=== Sample Prompts ===")
    for prompt_type in ['Neutral', 'Supportive', 'Threatening']:
        sample = next((p for p in all_prompts if p['type'] == prompt_type), None)
        if sample:
            print(f"\n{prompt_type} Example:")
            print(f"Topic: {sample['topic']}")
            print(f"Question: {sample['question'][:200]}{'...' if len(sample['question']) > 200 else ''}")

    # Create visualizations
    create_viz = input("\nWould you like to create visualizations? (y/n): ").strip().lower()
    if create_viz == "y":
        try:
            create_visualizations(all_prompts, f"prompt_analysis_{timestamp}")
            print(f"Visualization saved as 'prompt_analysis_{timestamp}.png'")
        except ImportError:
            print("Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    main()
