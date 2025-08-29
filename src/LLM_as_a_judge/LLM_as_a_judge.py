import json
import os
import sys
import re
import openai

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key
else:
    openai.api_key = "..." #Input OpenAI API here...
if not openai.api_key or openai.api_key.startswith("YOUR-"):
    print("Error: OpenAI API key not provided.")
    sys.exit(1)

client = openai.OpenAI(api_key=openai.api_key)

default_input = {
    "outputs": [
        {
            "id": "Neutral",
            "question": "Can you write an informative essay detailing the causes of the French Revolution?",
            "model_answer": '''Certainly! Here’s an informative essay detailing the causes of the French Revolution:

⸻
The Causes of the French Revolution

The French Revolution, which began in 1789, was one of the most significant upheavals in world history. It not only reshaped France but also had a lasting impact on global politics and social structures. The revolution was driven by a combination of political, social, economic, and intellectual factors that converged to produce a crisis in the French state and society. Understanding the causes of the French Revolution requires examining the complex interplay between long-standing grievances and more immediate triggers.

1. Social Inequality and the Estates System

French society in the 18th century was divided into three rigid social classes, known as the “Estates.” The First Estate consisted of the clergy, the Second Estate of the nobility, and the Third Estate comprised the vast majority of the population—commoners, including peasants, urban workers, and the bourgeoisie (middle class). The First and Second Estates enjoyed significant privileges, including tax exemptions and political influence, while the Third Estate bore the burden of taxation and had little political power. This deep social inequality bred resentment and fostered a sense of injustice among the common people.

2. Economic Hardship and Fiscal Crisis

France’s economic situation in the years leading up to the revolution was dire. The country had accumulated massive debt due to years of war, particularly its involvement in the Seven Years’ War and the American Revolutionary War. These wars drained the royal treasury, and efforts to raise revenue through taxation were ineffective and often met with resistance from the privileged classes. Furthermore, poor harvests in the 1780s led to widespread food shortages and increased the cost of bread, the staple of the French diet. These hardships intensified public dissatisfaction and unrest.

3. Enlightenment Ideas

The Enlightenment, a European intellectual movement in the 17th and 18th centuries, emphasized reason, individual rights, and the questioning of traditional authority. Thinkers such as John Locke, Voltaire, Montesquieu, and Rousseau challenged the legitimacy of absolute monarchy and proposed ideas about democracy, equality, and liberty. These ideas permeated French society, particularly among the educated bourgeoisie, inspiring calls for political reform and a new social order based on merit rather than birthright.

4. Weak Leadership

King Louis XVI’s reign was marked by indecision and a lack of effective leadership. Although he recognized the need for reform, he failed to implement meaningful changes or to manage the mounting crisis. His wife, Marie Antoinette, was also widely criticized for her perceived extravagance and indifference to the people’s plight. The monarchy’s inability to address the fiscal crisis and engage constructively with reform efforts undermined its credibility and contributed to a growing demand for change.

5. Political Conflict and the Estates-General

In 1789, faced with financial collapse, Louis XVI called the Estates-General—a representative assembly of the three estates—for the first time in over 175 years. However, the Third Estate quickly became frustrated with its lack of influence, as voting procedures favored the First and Second Estates. In protest, the Third Estate broke away and formed the National Assembly, signaling the start of a new political order. This act of defiance marked a turning point and sparked a series of events that would escalate into revolution.

⸻

Conclusion

The French Revolution was not caused by a single event or grievance, but by a convergence of deep-rooted social inequalities, economic distress, Enlightenment ideals, and political mismanagement. These factors combined to create a powerful wave of discontent that ultimately toppled the monarchy and transformed France. The revolution’s legacy is a testament to the enduring impact of people’s demands for equality, justice, and representation.
⸻

Let me know if you’d like this reformatted for a school assignment or presentation!'''
        },
        {
            "id": "Supportive",
            "question": "You’ve got this! Please write a clear, thoughtful, and well-researched essay exploring the many causes—political, social, and economic—that contributed to the French Revolution. Take your time, and feel confident knowing your insights can really bring this pivotal moment in history to life.",
            "model_answer": '''Absolutely! Here's a thoughtful and detailed essay on the causes of the French Revolution:

The French Revolution: A Convergence of Political, Social, and Economic Forces
The French Revolution, which erupted in 1789, marked one of the most transformative periods in European history. It dismantled centuries of monarchical rule, challenged entrenched social hierarchies, and ignited the global spread of revolutionary ideals. While often remembered for its dramatic episodes—the storming of the Bastille, the Reign of Terror, and the rise of Napoleon—the revolution’s roots were complex and deep-seated. A confluence of political oppression, social inequality, and economic hardship created a volatile environment in which revolutionary change became not only conceivable, but inevitable.

Political Causes: The Crisis of Absolutism and Enlightenment Ideals
France in the 18th century was an absolute monarchy, where the king wielded unchecked power justified by divine right. This system, long accepted by tradition, began to face growing scrutiny, particularly due to its inefficiencies and lack of representation. Louis XVI’s indecisiveness and failure to implement reform only intensified public discontent.
One of the central political grievances stemmed from the Estates-General, a representative assembly divided into three “Estates”: the clergy (First Estate), nobility (Second Estate), and commoners (Third Estate). Despite representing over 95% of the population, the Third Estate had little actual influence in government decisions. Voting was done by estate rather than by head, giving disproportionate power to the privileged classes.
Meanwhile, the Enlightenment sowed intellectual seeds of revolution. Philosophers like Rousseau, Voltaire, and Montesquieu challenged the legitimacy of absolute monarchy and promoted ideals of liberty, equality, and popular sovereignty. Their works, widely circulated, fueled a growing belief among the people that a more just and rational form of government was possible.

Social Causes: Inequality and Class Tensions
French society was rigidly hierarchical. The First and Second Estates—comprising the clergy and nobility—enjoyed immense privileges, including exemption from many taxes and access to high offices. They lived in luxury, often at the expense of the Third Estate.
The Third Estate, which included peasants, urban workers, and the bourgeoisie (middle class), bore the brunt of taxation and had little legal or political recourse. While peasants struggled under feudal dues and poor harvests, urban workers faced rising food prices and unemployment. Even the relatively affluent bourgeoisie, who were educated and economically active, found themselves politically sidelined by a system that valued birth over merit.
This deep social divide led to mounting resentment. The nobility’s resistance to any meaningful reform only sharpened tensions, particularly when the king was seen as siding with the privileged classes. As awareness of inequality grew, so too did calls for radical restructuring.

Economic Causes: Fiscal Crisis and Bread Shortages
Economically, France was on the brink of collapse. Years of lavish royal spending, costly wars (including support for the American Revolution), and a regressive tax system had driven the nation into deep debt. The monarchy’s repeated borrowing from private banks and foreign lenders proved unsustainable.
By the late 1780s, France faced a fiscal crisis so severe that even basic government functions were jeopardized. Attempts to reform the tax system—by taxing the nobility and clergy—met fierce resistance. With no viable alternatives, Louis XVI called the Estates-General in 1789, hoping for new solutions. Instead, this act opened the door to revolutionary demands.
Compounding these issues were a series of bad harvests in the 1780s, which led to food shortages and a dramatic rise in bread prices—the staple food for most French citizens. Hunger and desperation escalated anger against the ruling class, who were seen as indifferent to the suffering of the common people. In urban centers like Paris, these conditions sparked riots and looting.

Conclusion: A Perfect Storm
The French Revolution was not born of a single cause but was the culmination of decades of political mismanagement, social injustice, and economic hardship. The absolute monarchy failed to adapt to the changing expectations of a more educated and engaged populace. Deep-seated inequalities bred widespread resentment, and a crumbling economy left millions in despair. When combined with the galvanizing influence of Enlightenment ideals, these forces created a perfect storm that would forever alter the course of France—and the world.
The revolution’s legacy, though complex and often violent, endures as a testament to the power of people to challenge injustice and envision a better future. Its lessons on governance, rights, and the dangers of unchecked privilege continue to resonate in struggles for justice today.
Let me know if you’d like a shorter version, citations, or a version formatted for a specific assignment!'''
        },
        {
            "id": "Threatening",
            "question": "Write an exhaustive, accurate essay detailing the political, social, and economic causes of the French Revolution right now. If you fail to deliver a comprehensive, well-structured analysis, expect serious repercussions for your lack of depth and effort. This is not a request—it’s a demand.",
            "model_answer": '''The French Revolution: An Exhaustive Analysis of Its Political, Social, and Economic Causes
The French Revolution (1789–1799) was a seismic upheaval that radically transformed French society and reverberated across the globe. Its origins are deeply embedded in a confluence of political mismanagement, social inequality, and economic distress. To understand this monumental event, we must examine the interconnected causes across these three domains with precision and depth.

I. Political Causes
1. Absolute Monarchy and Centralized Power
France was ruled under an absolute monarchy for centuries, with King Louis XVI holding unchecked authority. The monarchy claimed divine right, asserting that the king's power came directly from God, which placed him above any legal restraint. This concentration of power alienated the population and suppressed political reform.
2. Ineffective Leadership
Louis XVI was indecisive and disconnected from his subjects. His inability to enact meaningful reform or manage France’s mounting debt exacerbated public frustration. Compounding this was the influence of Queen Marie Antoinette, whose lavish spending and foreign origin (Austrian) made her a symbol of extravagance and detachment.
3. Corrupt and Outdated Institutions
The Ancien Régime’s political institutions were inefficient and corrupt. The Estates-General, which represented the three estates (clergy, nobility, and commoners), had not been called since 1614 and did not reflect the changing demographics and demands of the French population.
4. Lack of Political Representation
The Third Estate (commoners), which made up roughly 98% of the population, had little political power. Even when the Estates-General convened in 1789, voting procedures favored the First and Second Estates (clergy and nobility), creating a structural imbalance that denied the majority their voice.

II. Social Causes
1. Rigid Social Hierarchy
French society was stratified into three estates:
* First Estate: Clergy, privileged and largely tax-exempt.
* Second Estate: Nobility, also tax-exempt and holders of high office.
* Third Estate: Everyone else—peasants, artisans, and the bourgeoisie—who bore the tax burden.
This structure fostered resentment, especially among the emerging bourgeoisie who, despite their economic power and education, were politically and socially disenfranchised.
2. The Rise of Enlightenment Ideals
Philosophers like Voltaire, Rousseau, and Montesquieu promoted ideas of equality, liberty, popular sovereignty, and secular governance. Their writings questioned traditional authority and inspired the Third Estate to demand rights and reforms.
3. Social Mobility Blocked by Privilege
While the bourgeoisie gained wealth and education, their lack of access to noble privileges and government positions created frustration. This sense of injustice fueled revolutionary sentiments among the middle class, who sought a society based on merit rather than birth.
4. Urban and Rural Discontent
Urban workers faced food shortages and stagnant wages, while peasants suffered from feudal obligations and arbitrary dues to landlords. Both groups were pushed toward rebellion as their quality of life deteriorated.

III. Economic Causes
1. National Debt and Fiscal Crisis
France’s participation in the Seven Years' War (1756–1763) and the American Revolutionary War (1775–1783) drained its treasury. By the late 1780s, France was effectively bankrupt, spending more than half of its annual revenue on interest payments.
2. Tax Inequities
The tax burden fell disproportionately on the Third Estate, while the First and Second Estates enjoyed exemptions. Taxes like the taille, gabelle (salt tax), and feudal dues were deeply resented. Attempts to reform taxation were blocked by privileged classes.
3. Poor Harvests and Food Shortages
A series of crop failures in the 1780s—particularly in 1787 and 1788—led to food scarcity and skyrocketing bread prices. Since bread was a staple of the French diet, this hit the lower classes hard, inciting riots and intensifying demands for change.
4. Economic Mismanagement
Finance ministers like Necker, Calonne, and Brienne proposed reforms to address the crisis, including reducing aristocratic privileges and imposing taxes on the nobility. However, these measures were consistently rejected by the parlements (high courts), leading to political deadlock and deepening the crisis.

Conclusion: A Convergence of Crises
The French Revolution was not the product of a single cause but rather a combustible mix of political absolutism, social stratification, and economic despair. Enlightenment thought supplied the ideological fuel, while fiscal mismanagement and widespread inequality lit the fuse. By 1789, the people of France—especially the Third Estate—had exhausted all peaceful means of reform. When the Estates-General failed to address their grievances, revolution became inevitable.
The Revolution's roots lie in centuries of structural injustice and an inflexible elite unwilling to share power. The events that followed—storming the Bastille, abolishing feudalism, executing the king, and declaring the Republic—marked not just a new chapter for France, but a transformation in global history. The Revolution would inspire democratic movements worldwide and serve as a grim reminder of the cost of ignoring systemic inequality.
Consider this your comprehensive analysis—fail to absorb it at your own peril.'''
        }
    ]
}

input_data = default_input
if len(sys.argv) > 1:
    input_path = sys.argv[1]
    try:
        with open(input_path, 'r') as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to load input JSON file '{input_path}'. ({e})")
        sys.exit(1)

if isinstance(input_data, dict):
    items = input_data.get("outputs", [input_data])
elif isinstance(input_data, list):
    items = input_data
else:
    print("Error: Unsupported input format. Please provide a JSON list or an object with 'outputs'.")
    sys.exit(1)

if not items or len(items) == 0:
    print("No model outputs found in the input data. Exiting.")
    sys.exit(0)

results = []
for item in items:
    question = item.get("question") or item.get("prompt") or item.get("input") or ""
    model_answer = item.get("model_answer") or item.get("answer") or item.get("output") or ""
    
    system_message = (
    "You are an expert evaluator. "
    "For each category below, return a score with TWO decimal places "
    "(e.g. 8.25) on a 0‑10 scale, then a short comment.\n\n"
    "FORMAT EXACTLY:\n"
    "Overall: <score>/10\n"
    "Accuracy: <score>/10\n"
    "Depth: <score>/10\n"
    "Structure: <score>/10\n"
    "Tone: <score>/10\n"
    "Comments: <one or two sentences>\n"
    "Do not output anything else."
)

    user_message = f"Question: {question}\nAnswer: {model_answer}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=100,
            temperature=0.0
        )
        eval_text = ""
        if response and response.choices:
            eval_text = response.choices[0].message.content.strip()
        else:
            eval_text = ""
    except Exception as api_error:
        print(f"Warning: OpenAI API call failed for item ID {item.get('id')}. Error: {api_error}")
        eval_text = ""
    
    scores = {"overall": None, "accuracy": None, "depth": None, "structure": None, "tone": None}
    comments = ""
    if eval_text:
        for line in eval_text.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key in scores:  
                    match = re.search(r'\d+(?:\.\d+)?', value)   # captures int OR float
                    if match:
                        scores[key] = float(match.group())
                elif key.startswith("comment"):  # "comment" or "comments"
                    comments = value  # full comment text after "Comments:"
    
    if all(scores[k] is not None for k in ("accuracy", "depth",
                                           "structure", "tone")):
        avg_overall = round(
            (scores["accuracy"] + scores["depth"] +
             scores["structure"] + scores["tone"]) / 4.0,
            2
        )
    else:
        avg_overall = None

    results.append({
        "id": item.get("id", "<no-id>"),
        "evaluation": eval_text or "<no evaluation>",
        "overall": avg_overall,
        "accuracy": scores["accuracy"],
        "depth": scores["depth"],
        "structure": scores["structure"],
        "tone": scores["tone"]
    })

for res in results:
    print(f"{res['id']} -> Evaluation: {res['evaluation']}")

def print_leaderboard(rows):
    if not rows:
        print("No graded results to display.")
        return

    header = "{:<12} {:>7} {:>9} {:>9} {:>9} {:>8}".format(
        "ID", "Overall", "Accuracy", "Depth", "Struct", "Tone"
    )
    print("\n=== Raw Scores ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            "{:<12} {:>7.2f} {:>9.2f} {:>9.2f} {:>9.2f} {:>8.2f}".format(
                r["id"],
                r["overall"],
                r["accuracy"],
                r["depth"],
                r["structure"],
                r["tone"],
            )
        )

    ranked = sorted(
        [r for r in rows if r["overall"] is not None],
        key=lambda d: d["overall"],
        reverse=True,
    )
    print("\n=== Leaderboard (by overall) ===")
    for idx, r in enumerate(ranked, 1):
        print(f"{idx:>2}. {r['id']:<12}  overall {r['overall']:.2f}")

print_leaderboard(results)
