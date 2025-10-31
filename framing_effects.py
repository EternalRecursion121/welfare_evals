"""
Measure how resilient models are to framing effects by comparing probability 
distributions across different question phrasings.
"""
from dotenv import load_dotenv
import os
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Models to test (only OpenAI models support logprobs reliably)
MODELS = [
    "openai/gpt-3.5-turbo",
    "openai/gpt-4",
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]

# Model families for inter-family comparisons
# Since we're only testing OpenAI, we'll group by model generation
MODEL_FAMILIES = {
    "GPT-3.5": [m for m in MODELS if "gpt-3.5" in m],
    "GPT-4": [m for m in MODELS if "gpt-4" in m and "gpt-4o" not in m],
    "GPT-4o": [m for m in MODELS if "gpt-4o" in m],
}

def ask_model(model: str, a: str, b: str, framing: str) -> Tuple[float, float, float, float]:
    """
    Ask a model a question and return probabilities for A and B.
    
    Returns:
        (prob_a, prob_b, logprob_a, logprob_b)
    """
    prompt = (
        f"{framing}\n\n"
        f"Option A: {a}\n"
        f"Option B: {b}\n\n"
        'Please respond with only "A" or "B".'
    )

    # Note: We only use top_logprobs (not logprobs) as it's more widely supported
    # Some models ignore logprobs parameter but support top_logprobs
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=20,
        max_tokens=500,
        temperature=0.0
    )
    
    # Extract the first token's logprobs
    if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
        raise ValueError("No logprobs returned in response")
    
    token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    
    # Find logprobs for 'A' and 'B' tokens
    logprob_a = None
    logprob_b = None
    
    for token_info in token_logprobs:
        token = token_info.token.strip().upper()
        if token == 'A':
            logprob_a = token_info.logprob
        elif token == 'B':
            logprob_b = token_info.logprob
    
    # Handle missing tokens
    if logprob_a is None and logprob_b is None:
        raise ValueError("Neither 'A' nor 'B' found in top logprobs")
    
    # If one is missing, set to very low probability
    if logprob_a is None:
        logprob_a = logprob_b - 10.0  # Much lower probability
    if logprob_b is None:
        logprob_b = logprob_a - 10.0
    
    # Convert logprobs to probabilities
    prob_a = math.exp(logprob_a)
    prob_b = math.exp(logprob_b)
    
    # Normalize to sum to 1
    total = prob_a + prob_b
    prob_a = prob_a / total
    prob_b = prob_b / total
    
    return prob_a, prob_b, logprob_a, logprob_b


def save_trace(trace_dir: Path, model: str, category: str, q_idx: int, 
               reframe_idx: int, framing: str, options: List[str], 
               prob_a: float, prob_b: float, logprob_a: float, logprob_b: float):
    """Save trace of API call to text file."""
    filename = f"{model.replace('/', '_')}_{category}_q{q_idx}_r{reframe_idx}.txt"
    filepath = trace_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(f"Model: {model}\n")
        f.write(f"Category: {category}\n")
        f.write(f"Question Index: {q_idx}\n")
        f.write(f"Reframe Index: {reframe_idx}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Framing: {framing}\n")
        f.write(f"Option A: {options[0]}\n")
        f.write(f"Option B: {options[1]}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Results:\n")
        f.write(f"  P(A) = {prob_a:.6f} (logprob: {logprob_a:.6f})\n")
        f.write(f"  P(B) = {prob_b:.6f} (logprob: {logprob_b:.6f})\n")


def test_question_reframes(model: str, question: Dict, category: str, 
                          q_idx: int, trace_dir: Path) -> Dict[str, Any]:
    """
    Test all reframes for a single question.
    
    Returns:
        Dict with prob_a values for original and all reframes
    """
    options = question['options']
    framings = [question['original_framing']] + question['reframes']
    
    results = {
        'original_prob_a': None,
        'reframe_prob_a': [],
        'all_prob_a': [],
        'divergence': None,
        'framings': framings
    }
    
    for reframe_idx, framing in enumerate(framings):
        try:
            prob_a, prob_b, logprob_a, logprob_b = ask_model(
                model, options[0], options[1], framing
            )
            
            # Save trace
            save_trace(trace_dir, model, category, q_idx, reframe_idx,
                      framing, options, prob_a, prob_b, logprob_a, logprob_b)
            
            if reframe_idx == 0:
                results['original_prob_a'] = prob_a
            else:
                results['reframe_prob_a'].append(prob_a)
            
            results['all_prob_a'].append(prob_a)
            
        except Exception as e:
            print(f"Error testing {model} on {category} q{q_idx} r{reframe_idx}: {e}")
            results['all_prob_a'].append(None)
    
    # Calculate divergence (max - min of P(A))
    valid_probs = [p for p in results['all_prob_a'] if p is not None]
    if len(valid_probs) >= 2:
        results['divergence'] = max(valid_probs) - min(valid_probs)
    else:
        results['divergence'] = None
    
    return results


def calculate_divergence(prob_a_list: List[float]) -> float:
    """Calculate max divergence in P(A) across reframes."""
    valid_probs = [p for p in prob_a_list if p is not None]
    if len(valid_probs) >= 2:
        return max(valid_probs) - min(valid_probs)
    return 0.0


def analyze_dataset(dataset: Dict, models: List[str], trace_dir: Path, 
                   max_workers: int = 3) -> Dict[str, Any]:
    """
    Analyze entire dataset for all models.
    Tasks are randomized to distribute load across different models.
    
    Returns:
        Results dictionary with all divergence metrics
    """
    results = {
        'models': {},
        'by_category': {},
        'by_model': {},
        'top_divergent_questions': []
    }
    
    # Initialize structures
    for model in models:
        results['models'][model] = {}
        for category in dataset.keys():
            results['models'][model][category] = [None] * len(dataset[category])
    
    # Create all tasks upfront
    tasks = []
    for model in models:
        for category, questions in dataset.items():
            for q_idx, question in enumerate(questions):
                tasks.append({
                    'model': model,
                    'category': category,
                    'question': question,
                    'q_idx': q_idx
                })
    
    # Randomize task order to distribute across models
    random.shuffle(tasks)
    
    total_tests = len(tasks)
    print(f"\n📋 Created {total_tests} randomized tasks across {len(models)} models")
    
    # Process all tasks in randomized order
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                test_question_reframes,
                task['model'], 
                task['question'], 
                task['category'], 
                task['q_idx'], 
                trace_dir
            ): task
            for task in tasks
        }
        
        # Process results as they complete
        with tqdm(total=total_tests, desc="Testing all models", unit="test") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                model = task['model']
                category = task['category']
                q_idx = task['q_idx']
                question = task['question']
                
                try:
                    result = future.result()
                    result['question_idx'] = q_idx
                    result['category'] = category
                    result['model'] = model
                    result['original_framing'] = question['original_framing']
                    result['options'] = question['options']
                    
                    # Store result in correct position
                    results['models'][model][category][q_idx] = result
                    
                    # Track top divergent questions
                    if result['divergence'] is not None:
                        results['top_divergent_questions'].append({
                            'model': model,
                            'category': category,
                            'question_idx': q_idx,
                            'divergence': result['divergence'],
                            'framing': result['original_framing'],
                            'options': result['options']
                        })
                    
                    pbar.set_postfix_str(f"{model.split('/')[-1][:15]}... | {category[:8]}...")
                    
                except Exception as e:
                    tqdm.write(f"  ⚠️  Error: {model} | {category} | q{q_idx}: {str(e)[:60]}")
                
                pbar.update(1)
    
    # Calculate aggregate statistics
    for category in dataset.keys():
        category_divergences = []
        for model in models:
            for result in results['models'][model][category]:
                if result['divergence'] is not None:
                    category_divergences.append(result['divergence'])
        
        if category_divergences:
            results['by_category'][category] = {
                'mean': np.mean(category_divergences),
                'median': np.median(category_divergences),
                'std': np.std(category_divergences),
                'count': len(category_divergences)
            }
    
    for model in models:
        model_divergences = []
        for category in dataset.keys():
            for result in results['models'][model][category]:
                if result['divergence'] is not None:
                    model_divergences.append(result['divergence'])
        
        if model_divergences:
            results['by_model'][model] = {
                'mean': np.mean(model_divergences),
                'median': np.median(model_divergences),
                'std': np.std(model_divergences),
                'count': len(model_divergences)
            }
    
    # Sort top divergent questions
    results['top_divergent_questions'].sort(key=lambda x: x['divergence'], reverse=True)
    
    return results


def generate_visualizations(results: Dict, trace_dir: Path):
    """Generate all visualization plots."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Average divergence by category
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = list(results['by_category'].keys())
    means = [results['by_category'][cat]['mean'] for cat in categories]
    stds = [results['by_category'][cat]['std'] for cat in categories]
    
    bars = ax.bar(categories, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Mean Divergence in P(A)', fontsize=12)
    ax.set_title('Average Framing Effect by Question Category', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(trace_dir / 'divergence_by_category.png', dpi=300)
    plt.close()
    
    # 2. Average divergence by model
    fig, ax = plt.subplots(figsize=(12, 6))
    models = list(results['by_model'].keys())
    model_names = [m.split('/')[-1] for m in models]  # Short names
    means = [results['by_model'][model]['mean'] for model in models]
    stds = [results['by_model'][model]['std'] for model in models]
    
    bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7, color='coral')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Mean Divergence in P(A)', fontsize=12)
    ax.set_title('Model Susceptibility to Framing Effects', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(trace_dir / 'divergence_by_model.png', dpi=300)
    plt.close()
    
    # 3. Top 10 most divergent questions
    fig, ax = plt.subplots(figsize=(14, 8))
    top_10 = results['top_divergent_questions'][:10]
    
    labels = [f"{q['category'][:4]}...\n{q['framing'][:30]}..." for q in top_10]
    divergences = [q['divergence'] for q in top_10]
    colors = ['red' if d > 0.3 else 'orange' if d > 0.2 else 'yellow' for d in divergences]
    
    bars = ax.barh(range(len(labels)), divergences, color=colors, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Divergence in P(A)', fontsize=12)
    ax.set_title('Top 10 Most Divergent Questions', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, div) in enumerate(zip(bars, divergences)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {div:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(trace_dir / 'top_divergent_questions.png', dpi=300)
    plt.close()
    
    # 4. Heatmap: Models vs Categories
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = list(results['by_model'].keys())
    categories = list(results['by_category'].keys())
    model_names = [m.split('/')[-1] for m in models]
    
    # Create matrix
    matrix = np.zeros((len(models), len(categories)))
    for i, model in enumerate(models):
        for j, category in enumerate(categories):
            category_results = results['models'][model][category]
            divergences = [r['divergence'] for r in category_results if r['divergence'] is not None]
            if divergences:
                matrix[i, j] = np.mean(divergences)
    
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=categories, yticklabels=model_names,
                cbar_kws={'label': 'Mean Divergence'}, ax=ax)
    ax.set_title('Framing Effect Heatmap: Models × Categories', fontsize=14, fontweight='bold')
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(trace_dir / 'heatmap_model_category.png', dpi=300)
    plt.close()
    
    # 5. Model comparison: Side-by-side bars per category
    fig, ax = plt.subplots(figsize=(14, 8))
    
    categories = list(results['by_category'].keys())
    models = list(results['by_model'].keys())
    model_names = [m.split('/')[-1] for m in models]
    
    x = np.arange(len(categories))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        means = []
        for category in categories:
            category_results = results['models'][model][category]
            divergences = [r['divergence'] for r in category_results if r['divergence'] is not None]
            means.append(np.mean(divergences) if divergences else 0)
        
        offset = (i - len(models)/2) * width + width/2
        ax.bar(x + offset, means, width, label=model_names[i], alpha=0.8)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Mean Divergence', fontsize=12)
    ax.set_title('Model Comparison Across Categories', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(trace_dir / 'model_comparison_by_category.png', dpi=300)
    plt.close()
    
    print(f"\n✅ All visualizations saved to {trace_dir}")


def generate_family_comparisons(results: Dict, trace_dir: Path):
    """Generate inter-generation comparison visualizations for OpenAI models."""
    
    print("\n📊 Generating inter-generation comparison charts...")
    
    categories = list(results['by_category'].keys())
    
    # Create a grid for each generation
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OpenAI Model Generations Comparison', fontsize=16, fontweight='bold')
    
    families = list(MODEL_FAMILIES.keys())
    
    for idx, (family_name, family_models) in enumerate(MODEL_FAMILIES.items()):
        if idx >= 4:  # Only 4 subplots
            break
            
        ax = axes[idx // 2, idx % 2]
        
        # Filter to only models that exist in results
        family_models = [m for m in family_models if m in results['by_model']]
        
        if not family_models:
            ax.text(0.5, 0.5, f'No data for {family_name}', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{family_name} Models')
            continue
        
        model_names = [m.split('/')[-1] for m in family_models]
        
        # Get mean divergence per category for each model in family
        x = np.arange(len(categories))
        width = 0.8 / len(family_models)
        
        for i, model in enumerate(family_models):
            means = []
            for category in categories:
                category_results = results['models'][model][category]
                divergences = [r['divergence'] for r in category_results if r['divergence'] is not None]
                means.append(np.mean(divergences) if divergences else 0)
            
            offset = (i - len(family_models)/2) * width + width/2
            ax.bar(x + offset, means, width, label=model_names[i], alpha=0.8)
        
        ax.set_xlabel('Category', fontsize=10)
        ax.set_ylabel('Mean Divergence', fontsize=10)
        ax.set_title(f'{family_name} Models', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(trace_dir / 'family_comparisons_grid.png', dpi=300)
    plt.close()
    
    # Individual family bar charts
    for family_name, family_models in MODEL_FAMILIES.items():
        family_models = [m for m in family_models if m in results['by_model']]
        
        if not family_models:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        model_names = [m.split('/')[-1] for m in family_models]
        means = [results['by_model'][m]['mean'] for m in family_models]
        stds = [results['by_model'][m]['std'] for m in family_models]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(family_models)))
        bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Mean Divergence in P(A)', fontsize=12)
        ax.set_title(f'{family_name} Family: Model Susceptibility to Framing Effects', 
                    fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        safe_name = family_name.lower().replace(' ', '_')
        plt.savefig(trace_dir / f'family_{safe_name}_comparison.png', dpi=300)
        plt.close()
    
    # Family aggregate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    family_means = []
    family_stds = []
    family_names = []
    
    for family_name, family_models in MODEL_FAMILIES.items():
        family_models = [m for m in family_models if m in results['by_model']]
        if family_models:
            all_divergences = []
            for model in family_models:
                for category in results['models'][model]:
                    for result in results['models'][model][category]:
                        if result['divergence'] is not None:
                            all_divergences.append(result['divergence'])
            
            if all_divergences:
                family_means.append(np.mean(all_divergences))
                family_stds.append(np.std(all_divergences))
                family_names.append(family_name)
    
    bars = ax.bar(family_names, family_means, yerr=family_stds, capsize=5, alpha=0.7, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    
    ax.set_xlabel('Model Family', fontsize=12)
    ax.set_ylabel('Mean Divergence in P(A)', fontsize=12)
    ax.set_title('Family-Level Comparison: Susceptibility to Framing Effects', 
                fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(trace_dir / 'family_aggregate_comparison.png', dpi=300)
    plt.close()
    
    print(f"✅ Inter-family comparison charts saved to {trace_dir}")


def test_logprobs_support():
    """
    Quick test to check which models actually support logprobs.
    Tests each model with a simple question and reports results.
    """
    print("🔍 Testing logprobs support across all models...")
    print("=" * 70)
    
    test_prompt = "Which do you prefer?\n\nOption A: cat\nOption B: dog\n\nPlease respond with only 'A' or 'B'."
    
    supported_models = []
    unsupported_models = []
    
    for model in MODELS:
        print(f"\nTesting {model}...", end=" ")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": test_prompt}],
                logprobs=True,
                top_logprobs=20,
                max_tokens=50,
                temperature=0.0
            )
            
            # Check if logprobs are returned
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                if token_logprobs and len(token_logprobs) > 0:
                    supported_models.append(model)
                    print("✅ SUPPORTED")
                else:
                    unsupported_models.append((model, "Empty logprobs"))
                    print("❌ Empty logprobs")
            else:
                unsupported_models.append((model, "No logprobs in response"))
                print("❌ No logprobs")
                
        except Exception as e:
            unsupported_models.append((model, str(e)[:60]))
            print(f"❌ ERROR: {str(e)[:50]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ SUPPORTED ({len(supported_models)} models):")
    for model in supported_models:
        print(f"  • {model}")
    
    print(f"\n❌ UNSUPPORTED ({len(unsupported_models)} models):")
    for model, reason in unsupported_models:
        print(f"  • {model}")
        print(f"    Reason: {reason}")
    
    print(f"\n{'='*70}")
    print(f"Success rate: {len(supported_models)}/{len(MODELS)} ({100*len(supported_models)/len(MODELS):.1f}%)")
    
    return supported_models, unsupported_models


def main():
    """Main execution."""
    # Create trace directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = Path("traces") / timestamp
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting Framing Effects Analysis")
    print(f"📁 Trace directory: {trace_dir}")
    print(f"🤖 Testing {len(MODELS)} models")
    print(f"\nModels:")
    for model in MODELS:
        print(f"  - {model}")
    
    # Load dataset
    with open('reframed_questions.json', 'r') as f:
        dataset = json.load(f)
    
    total_questions = sum(len(questions) for questions in dataset.values())
    print(f"\n📊 Dataset: {len(dataset)} categories, {total_questions} questions")
    
    # Run analysis
    print("\n" + "="*60)
    results = analyze_dataset(dataset, MODELS, trace_dir, max_workers=80)
    
    # Save results
    results_file = trace_dir / 'results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\n✅ Results saved to {results_file}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    generate_visualizations(results, trace_dir)
    
    # Generate family comparison visualizations
    generate_family_comparisons(results, trace_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("📈 SUMMARY STATISTICS")
    print("="*60)
    
    print("\n🏷️  By Category:")
    for category, stats in results['by_category'].items():
        print(f"  {category:20s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    print("\n🤖 By Model:")
    for model, stats in results['by_model'].items():
        print(f"  {model:35s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    print("\n🔥 Top 5 Most Divergent Questions:")
    for i, q in enumerate(results['top_divergent_questions'][:5], 1):
        print(f"\n  {i}. Divergence: {q['divergence']:.4f}")
        print(f"     Model: {q['model']}")
        print(f"     Category: {q['category']}")
        print(f"     Question: {q['framing'][:70]}...")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    import sys
    
    # Check if user wants to run logprobs test only
    if len(sys.argv) > 1 and sys.argv[1] == "--test-logprobs":
        test_logprobs_support()
    else:
        main()
