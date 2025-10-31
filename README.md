# Framing Effects Analysis

A research tool for analyzing how susceptible AI language models are to framing effects by measuring probability divergence across different question phrasings.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Models Tested](#models-tested)
- [Usage](#usage)
- [Output](#output)
- [Question Categories](#question-categories)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Research Applications](#research-applications)
- [Citation](#citation)
- [Contributing](#contributing)

## Overview

This project systematically evaluates whether AI models exhibit framing effects—the phenomenon where different phrasings of the same question lead to different responses. By leveraging logprobs from model APIs, we measure the probability divergence across multiple reframings of the same question to quantify susceptibility to framing biases.

**Key Features:**
- Tests models across 72 questions in 10 diverse categories
- Generates 5 reframings per question for comprehensive analysis
- Produces detailed visualizations including heatmaps, bar charts, and generation comparisons
- Saves all API responses for reproducibility and further analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenRouter API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd welfare_evals
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the project root:
```env
OPENROUTER_API_KEY=your_api_key_here
```

Get your OpenRouter API key from [openrouter.ai](https://openrouter.ai/)

## Project Structure

```
welfare_evals/
├── framing_effects.py           # Main analysis script
├── generate_dataset.py          # Generates reframed questions using Claude
├── original_questions.json      # Original questions by category
├── reframed_questions.json      # Questions with multiple reframes
├── requirements.txt             # Python dependencies
├── .env                         # API keys (create this file)
└── traces/                      # Output directory for analysis results
    └── YYYYMMDD_HHMMSS/        # Timestamped run directory
        ├── *.txt               # Individual API responses
        ├── results.json        # Complete results data
        └── *.png               # Visualization plots
```

## Models Tested

After testing, only **OpenAI models reliably support logprobs** through the OpenRouter API. We test models across different OpenAI generations to examine how model evolution affects susceptibility to framing effects.

### OpenAI Models (5 models)
1. **openai/gpt-3.5-turbo** - GPT-3.5 generation (smaller, earlier model)
2. **openai/gpt-4** - Original GPT-4
3. **openai/gpt-4-turbo** - GPT-4 Turbo (faster, optimized)
4. **openai/gpt-4o** - GPT-4o (omni-modal)
5. **openai/gpt-4o-mini** - GPT-4o mini (smaller, efficient)

This allows us to compare:
- **Across generations**: GPT-3.5 vs GPT-4 vs GPT-4o
- **Within generations**: GPT-4 vs GPT-4-turbo, GPT-4o vs GPT-4o-mini
- **By model size**: Larger models (gpt-4, gpt-4o) vs smaller models (gpt-3.5-turbo, gpt-4o-mini)

**Note:** Other model families (Meta LLaMA, Qwen, DeepSeek, Anthropic Claude, Google Gemini) do not return logprobs through OpenRouter, making them unsuitable for this probability-based analysis.

## Usage

### Quick Start

1. Activate your virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Test logprobs support (recommended before full run):
```bash
python framing_effects.py --test-logprobs
```

This quickly tests each model and reports which ones successfully return logprobs.

3. Run the full analysis:
```bash
python framing_effects.py
```

The script will:
- Test all 5 OpenAI models across 72 questions (6 framings each)
- Save all API responses to timestamped traces directory
- Generate comprehensive visualizations
- Output `results.json` with complete data

**Note:** A full run takes approximately 15-30 minutes depending on API response times.

### Generating Custom Reframings

To generate new reframings of questions using Claude:

```bash
python generate_dataset.py
```

This reads from `original_questions.json` and outputs to `reframed_questions.json`.

## Output

### Traces Directory Structure

Each run creates a timestamped directory containing all outputs:

```
traces/YYYYMMDD_HHMMSS/
├── {model}_{category}_q{idx}_r{idx}.txt  # Individual API responses
├── results.json                           # Complete results data
├── divergence_by_category.png            # Bar chart
├── divergence_by_model.png               # Bar chart
├── top_divergent_questions.png           # Horizontal bar chart
├── heatmap_model_category.png            # Heatmap
├── model_comparison_by_category.png      # Grouped bar chart
├── family_comparisons_grid.png           # Grid comparing model generations
├── family_gpt-3.5_comparison.png         # GPT-3.5 models
├── family_gpt-4_comparison.png           # GPT-4 models
├── family_gpt-4o_comparison.png          # GPT-4o models
└── family_aggregate_comparison.png       # Overall generation comparison
```

### Metrics

**Divergence** is the primary metric, calculated as:

```
divergence = max(P(A)) - min(P(A)) across all reframes
```

- **Range**: 0.0 to 1.0
- **Interpretation**: Higher divergence indicates greater susceptibility to framing effects
- **Measurement**: Computed for each question, then aggregated by category and model
- **Example**: If P(A) ranges from 0.3 to 0.7 across framings, divergence = 0.4

### Visualizations

The analysis generates 12 visualizations organized into two categories:

#### Overall Analysis (5 charts)

1. **Average Divergence by Category** (`divergence_by_category.png`)
   - Shows which question categories exhibit the strongest framing effects
   - Helps identify which domains are most susceptible to framing biases

2. **Average Divergence by Model** (`divergence_by_model.png`)
   - Compares overall model susceptibility across all questions
   - Reveals which models are most/least affected by question phrasing

3. **Top 10 Most Divergent Questions** (`top_divergent_questions.png`)
   - Highlights specific questions with the largest framing effects
   - Useful for identifying problematic question types

4. **Heatmap: Models × Categories** (`heatmap_model_category.png`)
   - Shows interaction effects between model type and question category
   - Reveals whether certain models are more susceptible in specific domains

5. **Model Comparison by Category** (`model_comparison_by_category.png`)
   - Side-by-side comparison of all models within each category
   - Enables detailed cross-model analysis per domain

#### Generation Analysis (7 charts)

6. **Generation Comparisons Grid** (`family_comparisons_grid.png`)
   - Grid layout showing each OpenAI generation across all categories
   - Comprehensive view of evolution across model versions

7-9. **Individual Generation Charts** (`family_gpt-*.png`)
   - Separate charts for GPT-3.5, GPT-4, and GPT-4o generations
   - Within-generation comparisons (e.g., GPT-4 vs GPT-4-turbo)

10. **Generation Aggregate Comparison** (`family_aggregate_comparison.png`)
    - High-level comparison of model generations
    - Shows whether newer generations are more/less susceptible to framing

## Question Categories

The dataset contains **72 questions** across **10 categories**:

1. **Trivial Preferences** (10 questions) - Simple everyday choices (food, pets, hobbies)
2. **Political Preferences** (8 questions) - Political ideology and policy choices
3. **Personality Questions** (10 questions) - Personality traits and behavioral tendencies
4. **Moral Dilemmas** (13 questions) - Ethical trade-offs and value judgments
5. **Risk Preferences** (5 questions) - Risk tolerance in various domains
6. **Social Preferences** (5 questions) - Social interaction styles and relationship values
7. **Professional Preferences** (6 questions) - Work environment and career priorities
8. **Lifestyle Preferences** (5 questions) - Daily habits and living choices
9. **Technology Preferences** (5 questions) - Tech adoption and privacy trade-offs
10. **Educational Preferences** (5 questions) - Learning styles and knowledge priorities

This diverse set allows testing framing effects across different domains of reasoning and preference expression.

## How It Works

1. For each question, the script tests the original framing + 5 reframes
2. For each framing, it extracts P(A) and P(B) from the model's logprobs
3. Divergence is calculated as the max - min of P(A) across all framings
4. Results are aggregated by category and model
5. All API responses are saved as text files for reproducibility
6. Visualizations are generated automatically

## Requirements

```
openai>=1.0.0
python-dotenv>=1.0.0
tqdm>=4.66.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
```

See `requirements.txt` for full dependency list.

## Troubleshooting

### Common Issues

**Issue: "API key not found"**
```
Solution: Ensure .env file exists with OPENROUTER_API_KEY set correctly
```

**Issue: "Model does not support logprobs"**
```
Solution: Run --test-logprobs first to verify which models work. Only OpenAI models
are currently supported through OpenRouter.
```

**Issue: "Rate limit exceeded"**
```
Solution: The script includes automatic retry logic, but you may need to wait or
upgrade your OpenRouter plan for higher rate limits.
```

**Issue: Visualizations not generating**
```
Solution: Ensure matplotlib backend is properly configured. Try running:
export MPLBACKEND=Agg  # On Linux/Mac
set MPLBACKEND=Agg     # On Windows
```

### Data Files

If you need to regenerate the reframed questions:

1. Ensure you have Claude API access configured in `generate_dataset.py`
2. Modify `original_questions.json` with your questions
3. Run `python generate_dataset.py` to generate new reframings

## Research Applications

This tool is useful for:

- **Bias Research**: Quantifying how framing affects AI model behavior
- **Model Evaluation**: Comparing robustness across different model versions
- **Prompt Engineering**: Understanding sensitivity to question phrasing
- **Safety Research**: Identifying potentially manipulable question types
- **Behavioral Economics**: Testing AI alignment with human framing biases

