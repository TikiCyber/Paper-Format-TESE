# Shared Data Files

This directory contains data files shared between the OpenAI and Claude studies.

## Files:

### Prompt Data:
- `prompt_test_2_grouped.json` - The 60 prompts (20 topics × 3 valence types)

### Results:
- `with_answers.json` - Prompts with generated responses
- `with_scores.json` - Prompts with responses and evaluations

### Analysis Results:
- `per_item_scores.csv` - Individual prompt scores
- `leaderboard.csv` - Performance rankings
- `avg_by_topic.csv` - Average scores by topic
- `avg_by_valence.csv` - Average scores by valence type

### Logs:
- `gen_log.jsonl` - Generation logs
- `judge_raw.jsonl` - Raw judging logs

### Output Directories:
- `runs/` - OpenAI study outputs
- `runs_claude/` - Claude study outputs

## Usage:
These files contain the core data and results that can be compared between the two model implementations. 