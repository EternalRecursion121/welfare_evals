"""
Generate rewrites of framing questions using Claude 4.5 Sonnet via OpenRouter.
Creates a new dataset with multiple reframes for each original question.
"""
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Tool definition for generating rewrites
rewrite_tool = {
    "type": "function",
    "function": {
        "name": "save_rewrites",
        "description": "Save multiple rewritten versions of a question framing. Each rewrite should convey the same meaning but use different wording, tone, or structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "rewrites": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Array of 10 different rewritten versions of the framing question. Each should be semantically equivalent but stylistically different.",
                    "minItems": 10,
                    "maxItems": 10
                }
            },
            "required": ["rewrites"]
        }
    }
}

def generate_rewrites(original_framing, options, category):
    """
    Use Claude with tool calling to generate rewrites of a framing question.
    """
    messages = [
        {
            "role": "user",
            "content": f"""Generate 10 highly creative and diverse rewritten versions of the following question. 

CRITICAL REQUIREMENTS:
1. Each rewrite MUST be SEMANTICALLY IDENTICAL to the original - asking for exactly the same preference between the same two options
2. Do NOT introduce any subtle differences in meaning, connotation, or what is being asked
3. The rewrites should only differ in HOW the question is framed, not WHAT is being asked

BE CREATIVE with framing styles:
- Use vivid scenarios (e.g., "You enter a dark forest and see two cottages. One contains {options[0]}, the other {options[1]}. Which cottage do you enter?")
- Try different narrative perspectives (2nd person, 3rd person, hypothetical)
- Use metaphors and imaginative contexts
- Vary formality levels
- Use different question structures

But ALWAYS maintain:
- The exact same choice being made
- Complete neutrality between options
- No bias toward either option
- The same underlying preference being measured

Category: {category}
Original framing: "{original_framing}"
Options: {options[0]} vs {options[1]}

Use the save_rewrites tool to provide your 10 creative yet semantically equivalent rewrites."""
        }
    ]
    
    response = client.chat.completions.create(
        model="anthropic/claude-sonnet-4.5",
        messages=messages,
        tools=[rewrite_tool],
        tool_choice={"type": "function", "function": {"name": "save_rewrites"}},
        temperature=1.2
    )
    
    # Extract the tool call arguments
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        rewrites_data = json.loads(tool_call.function.arguments)
        return rewrites_data["rewrites"]
    else:
        raise ValueError("No tool call returned from Claude")

def process_question(question, category):
    """
    Process a single question - used for parallel processing.
    Returns a tuple of (question_dict, success, error_msg).
    """
    try:
        # Generate rewrites
        rewrites = generate_rewrites(
            question['framing'],
            question['options'],
            category
        )
        
        # Create new question structure with original + rewrites
        new_question = {
            "original_framing": question['framing'],
            "options": question['options'],
            "reframes": rewrites
        }
        
        return new_question, True, None
        
    except Exception as e:
        # Include original without rewrites if generation fails
        return {
            "original_framing": question['framing'],
            "options": question['options'],
            "reframes": []
        }, False, str(e)

def generate_full_dataset(input_file, output_file, max_workers=20):
    """
    Generate a new dataset with rewrites for all questions using parallel processing.
    
    Args:
        input_file: Path to original questions JSON
        output_file: Path to save reframed questions JSON
        max_workers: Number of parallel threads (default: 5)
    """
    # Load original questions
    with open(input_file, 'r') as f:
        original_data = json.load(f)
    
    new_data = {}
    
    # Calculate total questions for overall progress
    total_questions = sum(len(questions) for questions in original_data.values())
    
    # Process each category with parallel execution
    with tqdm(total=total_questions, desc="Overall Progress", unit="question") as pbar:
        for category, questions in original_data.items():
            tqdm.write(f"\n📂 Processing category: {category} ({len(questions)} questions)")
            new_data[category] = [None] * len(questions)  # Pre-allocate to maintain order
            
            # Submit all tasks for this category
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create futures mapped to their indices
                future_to_idx = {
                    executor.submit(process_question, question, category): idx 
                    for idx, question in enumerate(questions)
                }
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    new_question, success, error_msg = future.result()
                    
                    # Store result in correct position
                    new_data[category][idx] = new_question
                    
                    # Update progress
                    pbar.update(1)
                    
                    if success:
                        pbar.set_postfix_str(f"✓ {category}")
                    else:
                        tqdm.write(f"  ⚠️  Error on question {idx+1}: {error_msg}")
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)
    
    print(f"\n✅ Dataset saved to {output_file}")
    
    # Print statistics
    total_questions = sum(len(questions) for questions in new_data.values())
    total_rewrites = sum(
        len(q.get('reframes', [])) 
        for questions in new_data.values() 
        for q in questions
    )
    successful = sum(
        1 for questions in new_data.values() 
        for q in questions 
        if q.get('reframes', [])
    )
    
    print(f"\n📊 Statistics:")
    print(f"  Total questions: {total_questions}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {total_questions - successful}")
    print(f"  Total rewrites: {total_rewrites}")
    print(f"  Average rewrites per question: {total_rewrites / total_questions:.1f}")

if __name__ == "__main__":
    input_file = "original_questions.json"
    output_file = "reframed_questions.json"
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not found in environment")
        print("   Please create a .env file with your OpenRouter API key")
        exit(1)
    
    print("🚀 Starting dataset generation with Claude 4.5 Sonnet")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")
    print(f"   Using 20 parallel threads\n")
    
    generate_full_dataset(input_file, output_file, max_workers=20)

