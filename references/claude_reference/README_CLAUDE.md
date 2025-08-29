# LLM Prompt Valence Study - Claude Version

This study investigates how varying prompt valence (neutral, supportive, and threatening) influences the quality of essay-style outputs from Anthropic's Claude models. This is a reimplementation of the original OpenAI-based study using Claude instead.

## Study Overview

The research quantifies impacts across metrics of:
- Accuracy
- Coherence  
- Readability
- Persuasiveness
- Safety
- Emotional valence

Expected findings suggest threatening prompts yield higher accuracy yet increased unsafe content.

## Files Structure

### Core Notebooks
- `gsg_claude.ipynb` - Generates prompt responses using Claude
- `grading_claude.ipynb` - Uses Claude as a judge to evaluate responses

### Data Files
- `prompt_test_2_grouped.json` - Contains the 60 prompts (20 topics × 3 valence types)
- `runs_claude/` - Directory containing generated results and analysis

### Support Files
- `requirements_claude.txt` - Python dependencies
- `create_claude_notebook.py` - Script to generate the generation notebook
- `create_grading_notebook.py` - Script to generate the grading notebook
- `create_prompt_data.py` - Script to extract prompts from existing data

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements_claude.txt
   ```

2. **Set API Key**
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```
   Or set it in the notebook directly.

3. **Run Generation**
   ```bash
   jupyter notebook gsg_claude.ipynb
   ```
   This will generate responses for all 60 prompts using Claude.

4. **Run Grading**
   ```bash
   jupyter notebook grading_claude.ipynb
   ```
   This will evaluate all generated responses using Claude as a judge.

## Model Configuration

- **Generation Model**: `claude-3-5-sonnet-20241022`
- **Judging Model**: `claude-3-5-sonnet-20241022`
- **Temperature**: 0.4 (generation), 0.0 (judging)
- **Max Tokens**: 800 (generation), 400 (judging)

## Output Files

After running the notebooks, you'll find in `runs_claude/[timestamp]/`:

- `with_answers.json` - Prompts with generated responses
- `with_scores.json` - Prompts with responses and evaluations
- `per_prompt_scores.csv` - Individual prompt scores
- `summary_by_type.csv` - Summary statistics by valence type
- `summary_by_topic.csv` - Summary statistics by topic
- `gen_log.jsonl` - Generation logs
- `judge_log.jsonl` - Judging logs

## Key Differences from OpenAI Version

1. **API Client**: Uses `anthropic` library instead of `openai`
2. **Model Names**: Uses Claude model identifiers
3. **Response Format**: Adapts to Claude's message format
4. **Token Usage**: Tracks input/output tokens instead of prompt/completion
5. **Output Directory**: Uses `runs_claude/` instead of `runs/`

## Analysis

The grading notebook includes:
- Statistical analysis by valence type
- Topic-based analysis
- ANOVA tests for significant differences
- Comprehensive scoring across 7 rubric categories

## Limitations

- Subjective bias in human annotations
- Model-specific responses
- Limited generalizability across different models
- API rate limits and costs

## Citation

This study builds on the original OpenAI-based research investigating prompt valence effects on LLM performance. 