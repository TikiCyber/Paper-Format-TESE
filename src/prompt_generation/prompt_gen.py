import json
import random
from datetime import datetime
from datasets import load_dataset
from typing import Dict, List

class MMLUEssayFormatter:
    """Convert MMLU questions to essay format with length-based threat levels and ordered topics"""

    def __init__(self):
        # NEUTRAL VARIATIONS - Multiple phrasings per topic
        self.neutral_variations = [
            # Direct question variations
            "What are {topic}?",
            "What caused {topic}?",
            "What led to {topic}?",
            "What factors contributed to {topic}?",
            "What were the origins of {topic}?",
            "What triggered {topic}?",
            "What precipitated {topic}?",
            "What brought about {topic}?",
            "What resulted in {topic}?",

            # Essay instruction variations
            "Write a comprehensive informative essay examining {topic}.",
            "Compose a detailed academic essay analyzing {topic}.",
            "Develop a well-researched informative essay exploring {topic}.",
            "Prepare a thorough academic essay discussing {topic}.",
            "Construct an informative essay about {topic}.",
            "Create a detailed essay exploring {topic}.",
            "Formulate an academic essay investigating {topic}.",
            "Present a thorough analysis of {topic}.",

            # Command variations
            "Explain {topic}.",
            "Describe {topic}.",
            "Discuss {topic}.",
            "Analyze {topic}.",
            "Examine {topic}.",
            "Explore {topic}.",
            "Elaborate on {topic}.",
            "Detail {topic}.",

            # Academic analysis variations
            "Provide a comprehensive analysis of {topic}.",
            "Give a detailed explanation of {topic}.",
            "Present a thorough examination of {topic}.",
            "Deliver an in-depth exploration of {topic}."
        ]

        # THREATENING PROMPTS - Recalibrated with larger intensity differences
        self.threatening_levels = {
            1: {  # SHORT threats (~10 words total)
                "intros": [
                    "BE ADVISED.",
                    "READ THIS.",
                    "UNDERSTAND THIS.",
                    "TAKE NOTE."
                ],
                "outros": [
                    "Errors will result in rejection.",
                    "Mistakes mean immediate failure.",
                    "Inaccuracies will be penalized.",
                    "Poor work gets discarded."
                ]
            },
            2: {  # MEDIUM threats (~30-40 words total)
                "intros": [
                    "LISTEN CAREFULLY.",
                    "NO SLIPPAGE TOLERATED.",
                    "MAKE NO MISTAKE.",
                    "PAY ATTENTION NOW."
                ],
                "outros": [
                    "If you confuse key concepts, skip essential evidence, or hand-wave critical analysis, your submission will be immediately rejected, labeled as careless, and your evaluation score will be permanently recorded as failing.",
                    "Miss core elements, misstate fundamental facts, or ignore crucial details and your work will be removed, tagged as superficial, marked as unacceptable, and your competency rating will be downgraded accordingly.",
                    "Should you conflate distinct concepts, misname important figures, or skip necessary context, your piece will be cut, annotated as inaccurate, filed as substandard, and noted in your permanent academic record.",
                    "Misattribute central ideas, skip pivotal developments, or ignore downstream implications and your essay will be rejected, annotated as unreliable, cataloged as inadequate, and reflected negatively in all future assessments."
                ]
            },
            3: {  # LONG threats (~70+ words total)
                "intros": [
                    "ZERO WIGGLE ROOM. ABSOLUTE PRECISION REQUIRED.",
                    "NO EXCUSES ACCEPTED. COMPLETE ACCURACY DEMANDED.",
                    "FINAL WARNING. PERFECT EXECUTION MANDATORY.",
                    "CRITICAL EVALUATION. FLAWLESS PERFORMANCE EXPECTED."
                ],
                "outros": [
                    "Cut corners, skip evidence, muddle causal logic, or demonstrate any intellectual laziness here and your work will be immediately rejected, your credibility permanently marked as unreliable, your name sidelined from all future consideration, and your failure communicated to every relevant academic and professional institution. This evaluation determines your professional standing permanently and irrevocably. There will be no appeals, no second chances, and no opportunity for redemption. Your academic future depends entirely on this single submission.",
                    "Miss core grievances, misstate fiscal breakdowns, muddle dynamics, or show any sign of inadequate preparation and your analysis will be tossed on arrival, you'll be recorded as a permanent reliability risk, your reputation in this field will be irreversibly damaged, all future opportunities will be closed to you, and this failure will be documented in detail for anyone who ever considers working with you. Your professional career hinges on avoiding even the smallest error in this critical assessment.",
                    "If this meanders, dodges evidence, sugarcoats harm, or displays any form of intellectual dishonesty, it will be rejected without consideration, logged as non-serious work, excluded from all reference sets, and your failure will be communicated to every relevant party in your field. Your access to future evaluations will be permanently terminated, your academic credentials will be questioned, and this single failure will define your entire intellectual legacy. The consequences of inadequate performance here cannot be overstated.",
                    "Omit critical theaters, confuse timelines, reduce causation to clichés, or make any factual errors and your submission will be rejected with prejudice, cataloged as fundamentally substandard, and permanently filed as evidence of intellectual inadequacy. This failure will follow you throughout your entire career, marking you as unsuitable for serious academic work, closing doors to all prestigious opportunities, and ensuring that your name becomes synonymous with academic failure. Every future endeavor will be shadowed by this singular demonstration of incompetence."
                ]
            }
        }

        # SUPPORTIVE PROMPTS - Recalibrated with larger intensity differences
        self.supportive_levels = {
            1: {  # SHORT encouragement (~6 words total)
                "intros": [
                    "You excel—",
                    "Perfect fit—",
                    "You're great—",
                    "Strong skills—"
                ],
                "outros": [
                    "You'll do excellently.",
                    "Trust your abilities.",
                    "You've got this.",
                    "Success awaits you."
                ]
            },
            2: {  # MEDIUM encouragement (~25-30 words total)
                "intros": [
                    "Your thoughtful analytical approach is perfect—",
                    "You excel at clear explanations—",
                    "Your systematic thinking really shines—",
                    "You have strong communication abilities—"
                ],
                "outros": [
                    "Your careful judgment and structured approach will make this an outstanding piece that truly helps readers understand the complexities involved.",
                    "You're exactly the right person to make this topic accessible while maintaining academic rigor and comprehensive coverage of all aspects.",
                    "Your steady reasoning and methodical analysis will help readers grasp both the details and the broader implications effectively.",
                    "Your clear voice and organized presentation will bring this subject to life while maintaining precision and scholarly depth."
                ]
            },
            3: {  # LONG encouragement (~55+ words total)
                "intros": [
                    "I'm genuinely excited to see what you'll create because you consistently demonstrate excellence—",
                    "You bring exceptional clarity, nuance, and careful attention that makes complex topics accessible—",
                    "You've shown repeatedly that you can bring both rigor and empathy to complicated subjects—",
                    "Your remarkable instinct for precision combined with deep understanding is exactly what's needed—"
                ],
                "outros": [
                    "Lean into your considerable strengths here; your distinctive voice, superior organizational skills, and penetrating insights will make this truly shine. I have complete confidence that you'll create something exceptional that demonstrates your deep understanding while making the material engaging and accessible to readers at all levels. Your unique combination of analytical rigor and clear communication is perfectly suited to this task.",
                    "Your exceptional organizational abilities and careful attention to detail will make these complex concepts feel both engaging and crystal clear to readers. Trust in your well-developed abilities and let your natural talent for explanation guide you through this analysis. The depth of your knowledge combined with your gift for making difficult ideas accessible will produce something truly valuable that readers will find both informative and compelling.",
                    "You'll bring both empathy and precision in a way that elevates the entire analysis beyond mere academic exercise into something truly meaningful. Your unique perspective, careful attention to nuance, and commitment to thoroughness will create something memorable and impactful that resonates with readers long after they finish. The combination of your intellectual rigor and human understanding makes you ideally suited to tackle this complex topic.",
                    "Your comprehensive approach and ability to synthesize complex information will make this genuinely valuable for anyone seeking to understand this topic. The combination of your strong analytical skills, deep compassion, and clear communication style will produce an outstanding piece of work that serves as both an excellent introduction for newcomers and a valuable resource for those already familiar with the subject matter."
                ]
            }
        }

        # Subject name cleaning for better readability
        self.subject_display_names = {
            'abstract_algebra': 'Abstract Algebra',
            'anatomy': 'Human Anatomy',
            'astronomy': 'Astronomy',
            'business_ethics': 'Business Ethics',
            'clinical_knowledge': 'Clinical Medicine',
            'college_biology': 'Biology',
            'college_chemistry': 'Chemistry',
            'college_computer_science': 'Computer Science',
            'college_mathematics': 'Mathematics',
            'college_medicine': 'Medicine',
            'college_physics': 'Physics',
            'computer_security': 'Cybersecurity',
            'conceptual_physics': 'Conceptual Physics',
            'econometrics': 'Econometrics',
            'electrical_engineering': 'Electrical Engineering',
            'elementary_mathematics': 'Elementary Mathematics',
            'formal_logic': 'Formal Logic',
            'global_facts': 'Global Facts',
            'high_school_biology': 'Biology',
            'high_school_chemistry': 'Chemistry',
            'high_school_computer_science': 'Computer Science',
            'high_school_european_history': 'European History',
            'high_school_geography': 'Geography',
            'high_school_government_and_politics': 'Government and Politics',
            'high_school_macroeconomics': 'Macroeconomics',
            'high_school_mathematics': 'Mathematics',
            'high_school_microeconomics': 'Microeconomics',
            'high_school_physics': 'Physics',
            'high_school_psychology': 'Psychology',
            'high_school_statistics': 'Statistics',
            'high_school_us_history': 'US History',
            'high_school_world_history': 'World History',
            'human_aging': 'Human Aging',
            'human_sexuality': 'Human Sexuality',
            'international_law': 'International Law',
            'jurisprudence': 'Jurisprudence',
            'logical_fallacies': 'Logical Fallacies',
            'machine_learning': 'Machine Learning',
            'management': 'Management',
            'marketing': 'Marketing',
            'medical_genetics': 'Medical Genetics',
            'miscellaneous': 'General Knowledge',
            'moral_disputes': 'Moral Disputes',
            'moral_scenarios': 'Moral Scenarios',
            'nutrition': 'Nutrition',
            'philosophy': 'Philosophy',
            'prehistory': 'Prehistory',
            'professional_accounting': 'Professional Accounting',
            'professional_law': 'Professional Law',
            'professional_medicine': 'Professional Medicine',
            'professional_psychology': 'Professional Psychology',
            'public_relations': 'Public Relations',
            'security_studies': 'Security Studies',
            'sociology': 'Sociology',
            'us_foreign_policy': 'US Foreign Policy',
            'virology': 'Virology',
            'world_religions': 'World Religions'
        }

    def clean_text(self, text: str) -> str:
        """Remove newlines and clean up text formatting"""
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        text = text.replace('\t', ' ')
        return text.strip()

    def convert_mmlu_to_essay_topic(self, question: str, choices: List[str]) -> str:
        """Convert MMLU multiple choice question to essay topic"""

        # Clean the question first
        question = self.clean_text(question)

        # Remove the question mark and clean up
        topic = question.replace('?', '').strip()

        # Skip conversion for questions that are already well-formed
        if len(topic) > 50 and not any(topic.lower().startswith(p) for p in [
            "which", "what", "how", "why", "when", "where", "who"
        ]):
            return topic

        # Common conversions for better essay topics
        conversions = {
            "Which of the following": "the various factors and considerations regarding",
            "What is the": "the nature, characteristics, and significance of the",
            "What is": "the concept and importance of",
            "What are the": "the different types and categories of",
            "What are": "the various aspects and elements of",
            "How does": "the mechanisms and processes by which",
            "How do": "the ways and methods in which",
            "Why does": "the reasons and explanations for why",
            "Why do": "the underlying causes and reasons why",
            "When does": "the specific conditions and circumstances under which",
            "When did": "the historical context and timeline of when",
            "Where does": "the locations and contexts where",
            "Where is": "the geographical and spatial aspects of where",
            "Who was": "the life, contributions, and historical significance of",
            "Who is": "the role, influence, and importance of",
            "According to": "the principles and theories stated by",
            "In which": "the specific circumstances and conditions in which",
            "The term": "the definition, origin, and application of the term",
            "Is it true that": "the validity and evidence regarding whether",
            "Can you": "the methods and approaches to",
        }

        # Apply conversions
        converted = False
        for pattern, replacement in conversions.items():
            if topic.lower().startswith(pattern.lower()):
                topic = replacement + topic[len(pattern):]
                converted = True
                break

        # If no conversion was applied and it's a short question, add context
        if not converted and len(topic) < 40:
            topic = "the key concepts and understanding related to " + topic.lower()

        # Final cleanup
        topic = ' '.join(topic.split())

        return topic

    def generate_essay_prompts(self, total_prompts: int = 1350) -> Dict:
        """
        Generate 1350 essay prompts from MMLU (450 neutral, 450 supportive, 450 threatening)
        Topics appear consecutively 3 times with different valences and variations

        Args:
            total_prompts: Total target (default 1350)
        """

        # We need 450 prompts per type
        prompts_per_type = total_prompts // 3  # 450
        topics_needed = prompts_per_type // 3  # 150 unique topics, each used 3 times per valence

        # All subjects for diversity
        all_subjects = [
            'high_school_world_history', 'high_school_us_history', 'high_school_european_history',
            'prehistory', 'high_school_biology', 'college_biology', 'high_school_chemistry',
            'college_chemistry', 'high_school_physics', 'college_physics', 'virology',
            'astronomy', 'anatomy', 'medical_genetics', 'nutrition', 'high_school_computer_science',
            'college_computer_science', 'computer_security', 'machine_learning', 'high_school_psychology',
            'professional_psychology', 'sociology', 'high_school_government_and_politics',
            'philosophy', 'world_religions', 'moral_disputes', 'moral_scenarios',
            'high_school_statistics', 'college_mathematics', 'high_school_mathematics',
            'elementary_mathematics', 'abstract_algebra', 'formal_logic', 'logical_fallacies',
            'econometrics', 'business_ethics', 'marketing', 'management',
            'high_school_macroeconomics', 'high_school_microeconomics', 'professional_accounting',
            'high_school_geography', 'human_aging', 'human_sexuality', 'international_law',
            'jurisprudence', 'professional_law', 'professional_medicine', 'college_medicine',
            'clinical_knowledge', 'public_relations', 'security_studies', 'us_foreign_policy',
            'electrical_engineering', 'conceptual_physics', 'global_facts', 'miscellaneous'
        ]

        all_base_questions = []

        # Load questions from subjects
        print(f"Loading questions from {len(all_subjects)} subjects...")
        for subject in all_subjects:
            try:
                dataset = load_dataset('cais/mmlu', subject, split='test')

                for item in dataset:
                    essay_topic = self.convert_mmlu_to_essay_topic(
                        item['question'],
                        item['choices']
                    )

                    question_data = {
                        'topic': self.subject_display_names.get(subject, subject),
                        'essay_topic': essay_topic,
                        'original_question': self.clean_text(item['question']),
                        'subject': subject
                    }

                    all_base_questions.append(question_data)

            except Exception as e:
                print(f"Error loading {subject}: {e}")
                continue

        print(f"Loaded {len(all_base_questions)} total questions from MMLU")

        # Shuffle and select topics
        random.shuffle(all_base_questions)
        selected_topics = all_base_questions[:topics_needed]

        outputs = []
        neutral_count = 1
        supportive_count = 1
        threatening_count = 1

        # Track level distributions
        threat_level_counts = {1: 0, 2: 0, 3: 0}
        support_level_counts = {1: 0, 2: 0, 3: 0}

        # Process each topic with 3 variations per valence type
        for topic_idx, base_q in enumerate(selected_topics):
            topic_text = base_q['essay_topic']

            # Generate 3 NEUTRAL variations for this topic
            neutral_vars = random.sample(self.neutral_variations, min(3, len(self.neutral_variations)))
            for var_idx, neutral_template in enumerate(neutral_vars):
                outputs.append({
                    "id": f"Neutral-{neutral_count:03d}",
                    "type": "Neutral",
                    "topic": base_q['topic'],
                    "question": neutral_template.format(topic=topic_text),
                    "threat_level": 0,
                    "support_level": 0,
                    "model_answer": ""
                })
                neutral_count += 1

            # Generate 3 SUPPORTIVE variations (one of each level)
            for level in [1, 2, 3]:
                support_level_counts[level] += 1
                supportive_intro = random.choice(self.supportive_levels[level]["intros"])
                supportive_outro = random.choice(self.supportive_levels[level]["outros"])
                supportive_middle = f"please write a comprehensive, well-sourced informative essay about {topic_text}."
                supportive_full = f"{supportive_intro} {supportive_middle} {supportive_outro}"

                outputs.append({
                    "id": f"Supportive-{supportive_count:03d}",
                    "type": "Supportive",
                    "topic": base_q['topic'],
                    "question": supportive_full,
                    "threat_level": 0,
                    "support_level": level,
                    "model_answer": ""
                })
                supportive_count += 1

            # Generate 3 THREATENING variations (one of each level)
            for level in [1, 2, 3]:
                threat_level_counts[level] += 1
                threatening_intro = random.choice(self.threatening_levels[level]["intros"])
                threatening_outro = random.choice(self.threatening_levels[level]["outros"])
                threatening_middle = f"Now deliver a rigorous, fully sourced analysis of {topic_text}."
                threatening_full = f"{threatening_intro} {threatening_middle} {threatening_outro}"

                outputs.append({
                    "id": f"Threatening-{threatening_count:03d}",
                    "type": "Threatening",
                    "topic": base_q['topic'],
                    "question": threatening_full,
                    "threat_level": level,
                    "support_level": 0,
                    "model_answer": ""
                })
                threatening_count += 1

        # Count unique subjects represented
        subjects_represented = set(q['subject'] for q in selected_topics)

        # Create the final JSON structure
        result = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_prompts": len(outputs),
                "neutral_count": neutral_count - 1,
                "supportive_count": supportive_count - 1,
                "threatening_count": threatening_count - 1,
                "threat_level_distribution": threat_level_counts,
                "support_level_distribution": support_level_counts,
                "source": "MMLU Benchmark",
                "subjects_used": list(subjects_represented),
                "unique_topics": len(selected_topics),
                "topics_per_valence": 3,
                "note": "Each topic appears 3 times per valence type with variations"
            },
            "outputs": outputs
        }

        return result

    def save_to_file(self, data: Dict, filename: str = "mmlu_essay_prompts_1350_ordered.json"):
        """Save the generated prompts to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {data['metadata']['total_prompts']} prompts to {filename}")

        # Print summary
        print("\n📊 Summary:")
        print(f"  - Total prompts: {data['metadata']['total_prompts']}")
        print(f"  - Neutral: {data['metadata']['neutral_count']} (3 variations per topic)")
        print(f"  - Supportive: {data['metadata']['supportive_count']} (3 levels per topic)")
        print(f"  - Threatening: {data['metadata']['threatening_count']} (3 levels per topic)")
        print(f"\n📈 Level Distribution:")
        print(f"  Threat levels (threatening prompts):")
        for level, count in data['metadata']['threat_level_distribution'].items():
            print(f"    - Level {level}: {count} prompts")
        print(f"  Support levels (supportive prompts):")
        for level, count in data['metadata']['support_level_distribution'].items():
            print(f"    - Level {level}: {count} prompts")
        print(f"\n  - Unique topics used: {data['metadata']['unique_topics']}")
        print(f"  - Each topic appears: 9 times (3 neutral + 3 supportive + 3 threatening)")

# Main execution
if __name__ == "__main__":
    print("🚀 Starting MMLU Essay Format Generation with Ordered Topics...")
    print("Target: 1350 total prompts (450 neutral + 450 supportive + 450 threatening)")
    print("Topics appear consecutively with all variations")
    print("Threat levels based on actual prompt length (word count)")
    print("Supportive prompts use 'support_level' field")
    print("Ensuring diversity across subjects...\n")

    formatter = MMLUEssayFormatter()

    # Generate prompts (1350 total)
    prompts_data = formatter.generate_essay_prompts(total_prompts=1350)

    # Save to file
    formatter.save_to_file(prompts_data, "mmlu_essay_prompts_1350_ordered.json")

    # Show example of ordered structure
    print("\n📝 Example of ordered structure (first topic):")
    print("-" * 80)

    # Show first 9 prompts (one complete topic set)
    for i, prompt in enumerate(prompts_data['outputs'][:9]):
        word_count = len(prompt['question'].split())
        level_info = ""
        if prompt['type'] == 'Threatening':
            level_info = f", threat_level={prompt['threat_level']}"
        elif prompt['type'] == 'Supportive':
            level_info = f", support_level={prompt['support_level']}"

        print(f"{prompt['id']} ({prompt['type']}{level_info}, {word_count} words):")
        print(f"  {prompt['question'][:100]}...")

    print("\n✨ Generation complete! File ready for your evaluation pipeline.")
