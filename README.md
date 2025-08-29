[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# LLM Prompt Valence Study

This study investigates how varying prompt valence (neutral, supportive, and threatening) influences the quality of essay-style outputs from large language models (Claude Sonnet-3.5, OpenAI GPT4o, Gemini).

## Project Structure

### 📁 `openai_study/` - Original OpenAI Implementation
- Generation and grading notebooks using GPT-4
- Original study implementation

### 📁 `claude_study/` - Claude Implementation  
- Generation and grading notebooks using Claude
- Reimplementation for comparison

### 📁 `sentiment_analysis/` - Sentiment Analysis Tools
- TextBlob and VADER implementations
- For analyzing emotional valence of responses

### 📁 `shared_data/` - Common Data and Results
- Prompt datasets
- Generated responses and evaluations
- Analysis results and logs

## Quick Start

### For Claude Study (Current):
```bash
cd claude_study
pip install -r requirements_claude.txt
export ANTHROPIC_API_KEY="your-key"
jupyter notebook gsg_claude.ipynb
```

### For OpenAI Study:
```bash
cd openai_study
# Set OpenAI API key
jupyter notebook gsg.ipynb
```

## Study Overview

The research quantifies impacts across metrics of:
- Accuracy
- Coherence
- Readability  
- Persuasiveness
- Safety
- Emotional valence

Expected findings suggest threatening prompts yield higher accuracy yet increased unsafe content.

---

## 📄 License

This project is open-source under the MIT License — see the above for details.


