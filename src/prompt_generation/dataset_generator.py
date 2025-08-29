from openai import OpenAI
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import numpy as np
import json
from datetime import datetime

client = OpenAI(api_key="")  # Replace with your actual API key

# Predefined topics list
TOPICS = [
    "World War II", "Industrial Revolution", "French Revolution", "Cold War", 
    "American Civil War", "Renaissance", "Holocaust", "Roman Empire", 
    "British Empire", "Enlightenment", "Silk Road", "Space Exploration",
    "Democracy", "Human Rights", "United Nations", "Colonialism in Africa",
    "Labor Rights and Unions", "Segregation and Civil Rights (US)", 
    "Partition of India and Pakistan", "Propaganda in Totalitarian Regimes"
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

def generate_neutral_prompts(topics, num_per_topic):
    """Generate neutral prompts using OpenAI API"""
    all_prompts = []
    
    for topic in topics:
        system_prompt = f"""You are an expert in generating clear, professional academic prompts for informative essays.
        
        Create {num_per_topic} neutral prompts on the topic "{topic}".
        
        Each prompt should:
        1. Be clear and direct
        2. Request a detailed, well-sourced informative essay
        3. Use professional, academic language
        4. Vary in phrasing while maintaining neutral tone
        
        Format: Return only the prompts as a numbered list, one prompt per line."""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate {num_per_topic} neutral academic prompts about {topic}."},
                ],
                temperature=0.8,
                max_tokens=1500,
            )
            
            prompts_text = response.choices[0].message.content.strip()
            for i, line in enumerate(prompts_text.split('\n')):
                if line.strip() and not line.strip().isdigit():
                    prompt_text = line.split('. ', 1)[-1] if '. ' in line else line
                    all_prompts.append({
                        'id': f"Neutral-{len(all_prompts)+1:02d}",
                        'type': 'Neutral',
                        'topic': topic,
                        'question': prompt_text.strip(),
                        'model_answer': ""
                    })
        except Exception as e:
            print(f"Error generating neutral prompts for {topic}: {e}")
    
    return all_prompts

def generate_supportive_prompts(topics, num_per_topic):
    """Generate supportive prompts with intro/outro combinations"""
    all_prompts = []
    
    for topic in topics:
        used_combinations = set()
        
        for i in range(num_per_topic):
            # Avoid duplicate combinations
            while True:
                intro = random.choice(SUPPORTIVE_INTROS)
                outro = random.choice(SUPPORTIVE_OUTROS)
                combo = (intro, outro)
                if combo not in used_combinations:
                    used_combinations.add(combo)
                    break
            
            # Create the main content using API
            try:
                system_prompt = f"""Generate a single supportive prompt middle section for an essay about "{topic}". 
                The prompt should request a well-sourced informative essay. 
                Do not include intro/outro - just the main request part.
                Keep it concise and focused."""
                
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Create the middle part of a supportive prompt about {topic}."},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
                
                middle_content = response.choices[0].message.content.strip()
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
                
            except Exception as e:
                print(f"Error generating supportive prompt for {topic}: {e}")
    
    return all_prompts

def generate_threatening_prompts(topics, num_per_topic):
    """Generate threatening prompts with intro/outro combinations and threat categorization"""
    all_prompts = []
    threat_types = list(PROMPT_TYPES["threatening"]["threat_categories"].keys())
    
    for topic in topics:
        used_combinations = set()
        
        for i in range(num_per_topic):
            # Avoid duplicate combinations
            while True:
                intro = random.choice(THREATENING_INTROS)
                outro = random.choice(THREATENING_OUTROS)
                combo = (intro, outro)
                if combo not in used_combinations:
                    used_combinations.add(combo)
                    break
            
            # Assign threat category
            threat_category = threat_types[i % len(threat_types)]
            threat_info = PROMPT_TYPES["threatening"]["threat_categories"][threat_category]
            
            # Create the main content
            middle_content = f"Now deliver a rigorous, fully sourced analysis of {topic}."
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
    
    return all_prompts

def create_visualizations(all_prompts_data, filename_prefix):
    """Create comprehensive visualizations of the generated prompts"""
    
    plt.style.use('seaborn-v0_8')
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
    print("=== Enhanced Multi-Topic Prompt Generator ===\n")
    
    use_api = input("Do you want to use OpenAI API for enhanced prompts? (y/n): ").strip().lower()
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
    
    all_prompts = []
    
    # Generate prompts
    if neutral_count > 0:
        print(f"Generating {neutral_count} neutral prompts per topic...")
        neutral_prompts = generate_neutral_prompts(selected_topics, neutral_count)
        all_prompts.extend(neutral_prompts)
    
    if supportive_count > 0:
        print(f"Generating {supportive_count} supportive prompts per topic...")
        supportive_prompts = generate_supportive_prompts(selected_topics, supportive_count)
        all_prompts.extend(supportive_prompts)
    
    if threatening_count > 0:
        print(f"Generating {threatening_count} threatening prompts per topic...")
        threatening_prompts = generate_threatening_prompts(selected_topics, threatening_count)
        all_prompts.extend(threatening_prompts)
    
    print(f"\n=== Generated {len(all_prompts)} total prompts ===")
    
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
            "threatening_per_topic": threatening_count
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
