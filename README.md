# Framing Effects Analysis

This project analyzes how susceptible AI models are to framing effects by measuring probability divergence across different question phrasings.

## Files

- `framing_effects.py` - Main analysis script
- `generate_dataset.py` - Generates reframed questions using Claude
- `original_questions.json` - Original questions by category
- `reframed_questions.json` - Questions with multiple reframes
- `traces/` - Output directory for analysis results

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

### Test Logprobs Support
Before running the full analysis, test which models actually support logprobs:

```bash
cd welfare_evals
source venv/bin/activate
python framing_effects.py --test-logprobs
```

This will quickly test each model and report which ones successfully return logprobs.

### Run Full Analysis

```bash
cd welfare_evals
source venv/bin/activate
python framing_effects.py
```

## Output

### Traces Directory Structure
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

**Divergence**: max(P(A)) - min(P(A)) across all reframes of a question
- Higher divergence = more susceptible to framing effects
- Measured for each question, then aggregated by category and model

### Visualizations

#### Overall Comparisons
1. **Average Divergence by Category**: Shows which question categories (trivial, political, personality, moral) exhibit the strongest framing effects

2. **Average Divergence by Model**: Compares model susceptibility - which models are most/least affected by how questions are phrased

3. **Top 10 Most Divergent Questions**: Identifies specific questions where framing has the biggest impact on model responses

4. **Heatmap (Models × Categories)**: Shows the interaction between model type and question category

5. **Model Comparison by Category**: Side-by-side comparison of all models across each category

#### Inter-Generation Comparisons
6. **Generation Comparisons Grid**: Grid showing each OpenAI generation's performance across categories

7. **Individual Generation Charts**: Separate bar charts for each generation (GPT-3.5, GPT-4, GPT-4o) comparing models within that generation

8. **Generation Aggregate Comparison**: Overall comparison of model generations to see which generation is most/least susceptible to framing effects

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

See `requirements.txt` for dependencies.

