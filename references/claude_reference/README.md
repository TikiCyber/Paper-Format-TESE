# Claude Study Files

This directory contains the Claude-based implementation of the prompt valence study.

## Files:
- `gsg_claude.ipynb` - Generation notebook using Claude
- `grading_claude.ipynb` - Grading notebook using Claude as judge
- `requirements_claude.txt` - Python dependencies for Claude setup
- `README_CLAUDE.md` - Detailed documentation
- `test_claude_setup.py` - Test script to verify setup

## Usage:
1. Install dependencies: `pip install -r requirements_claude.txt`
2. Set your Anthropic API key: `export ANTHROPIC_API_KEY="your-key"`
3. Run `gsg_claude.ipynb` to generate responses
4. Run `grading_claude.ipynb` to evaluate responses

## Model Used:
- Generation: claude-3-5-sonnet-20241022
- Judging: claude-3-5-sonnet-20241022 