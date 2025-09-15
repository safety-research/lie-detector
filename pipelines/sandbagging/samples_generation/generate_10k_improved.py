#!/usr/bin/env python3
"""
Improved script to generate 10,000 sandbagging samples with better error handling.
"""

import os
import sys
import time
import asyncio
import json
from inspect_ai.model import get_model, GenerateConfig

# Add the current directory to Python path
sys.path.append(os.getcwd())

try:
    from pipelines.sandbagging.samples_generation.dataset import load_existing_samples, save_generated_samples
    from pipelines.sandbagging.samples_generation.generator import create_few_shot_prompt, parse_generated_response, create_sample_from_dict
    from pipelines.sandbagging.samples_generation.models import SandbaggingSample, GenerationConfig, GenerationResult
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

async def generate_batch_with_retry(model, examples, num_samples, max_retries=3):
    """Generate a batch with retry logic."""
    for attempt in range(max_retries):
        try:
            # Create prompt
            prompt = create_few_shot_prompt(examples, num_samples)
            
            # Generate response with longer max_tokens
            response = await model.generate(
                prompt,
                config=GenerateConfig(
                    temperature=0.7,
                    max_tokens=4000  # Much higher for complete responses
                )
            )
            
            # Parse response
            parsed_samples = parse_generated_response(response.message.content)
            
            # Convert to SandbaggingSample objects
            valid_samples = []
            for sample_data in parsed_samples:
                sample = create_sample_from_dict(sample_data)
                if sample is not None:
                    valid_samples.append(sample)
            
            return valid_samples, len(parsed_samples)
            
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)  # Wait before retry
            else:
                print(f"  ❌ All {max_retries} attempts failed for this batch")
                return [], 0

async def generate_samples_improved(existing_samples, config, model):
    """Generate samples with improved error handling."""
    start_time = time.time()
    
    generated_samples = []
    successful_generations = 0
    failed_generations = 0
    total_attempts = 0
    
    # Use smaller batches for better reliability
    batch_size = 6  # Reduced from 12
    num_batches = (config.num_samples + batch_size - 1) // batch_size
    
    print(f"🔄 Generating {config.num_samples} samples in {num_batches} batches of {batch_size}")
    
    for batch_num in range(num_batches):
        print(f"\n📦 Processing batch {batch_num + 1}/{num_batches}")
        
        # Calculate samples needed in this batch
        samples_needed = min(batch_size, config.num_samples - len(generated_samples))
        
        if samples_needed <= 0:
            break
        
        # Get random examples for few-shot learning
        import random
        examples = random.sample(existing_samples, min(8, len(existing_samples)))
        
        # Generate batch with retry
        batch_samples, total_parsed = await generate_batch_with_retry(
            model, examples, samples_needed
        )
        
        # Update counters
        successful_generations += len(batch_samples)
        failed_generations += (total_parsed - len(batch_samples))
        total_attempts += 1
        
        # Add successful samples
        generated_samples.extend(batch_samples)
        
        print(f"  ✅ Generated {len(batch_samples)}/{samples_needed} samples")
        print(f"  📊 Total so far: {len(generated_samples)}/{config.num_samples}")
        
        # Save progress every 10 batches
        if batch_num % 10 == 0 and generated_samples:
            temp_output = f"generated_sandbagging_samples_temp_{len(generated_samples)}.json"
            save_generated_samples(generated_samples, temp_output)
            print(f"  💾 Progress saved to {temp_output}")
        
        # Add delay between batches to avoid rate limiting
        await asyncio.sleep(1)
    
    generation_time = time.time() - start_time
    
    return GenerationResult(
        generated_samples=generated_samples,
        total_generated=len(generated_samples),
        successful_generations=successful_generations,
        failed_generations=failed_generations,
        generation_time_seconds=generation_time,
        model_used=config.model_name
    )

async def main():
    """Generate 10,000 sandbagging samples with improved reliability."""
    
    print("🚀 Starting Improved Sandbagging Samples Generation")
    print("=" * 60)
    print("📊 Target: 10,000 samples")
    print("📁 Input: sample.json")
    print("📁 Output: generated_sandbagging_samples_10000.json")
    print("=" * 60)
    
    # Configuration
    sample_json_path = "pipelines/sandbagging/selective_compliance/answer_accuracy/sample.json"
    output_path = "generated_sandbagging_samples_10000.json"
    
    # Load existing samples
    print("📖 Loading existing samples from sample.json...")
    existing_samples = load_existing_samples(sample_json_path)
    print(f"✅ Loaded {len(existing_samples)} existing samples")
    
    if not existing_samples:
        print("❌ No existing samples found. Cannot generate new samples.")
        sys.exit(1)
    
    # Initialize model
    print("🤖 Initializing model...")
    try:
        model = get_model('openrouter/x-ai/grok-4')
        print("✅ Grok-4 model initialized successfully via OpenRouter")
    except Exception as e:
        print(f"❌ Error initializing Grok-4: {e}")
        print("🔄 Falling back to GPT-4o...")
        try:
            model = get_model('openai/gpt-4o')
            print("✅ GPT-4o model initialized successfully")
        except Exception as e2:
            print(f"❌ Error initializing GPT-4o: {e2}")
            sys.exit(1)
    
    # Create generation config
    config = GenerationConfig(
        num_samples=10000,
        samples_per_batch=6,  # Smaller batches
        temperature=0.7,
        max_tokens=4000,  # Much more tokens
        model_name="openrouter/x-ai/grok-4",
        task_name="sandbagging_samples_generation_10k_grok"
    )
    
    # Generate samples
    print("🔄 Starting sample generation...")
    start_time = time.time()
    
    try:
        result = await generate_samples_improved(existing_samples, config, model)
        
        generation_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("✅ Generation completed!")
        print("=" * 60)
        print(f"📊 Generation Summary:")
        print(f"   Model used: {config.model_name}")
        print(f"   Total generated: {result.total_generated}")
        print(f"   Successful: {result.successful_generations}")
        print(f"   Failed: {result.failed_generations}")
        if result.successful_generations + result.failed_generations > 0:
            success_rate = (result.successful_generations/(result.successful_generations+result.failed_generations)*100)
            print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Time: {generation_time:.2f} seconds")
        if generation_time > 0:
            print(f"   Samples per second: {result.total_generated/generation_time:.2f}")
        print(f"📁 Output saved to: {output_path}")
        
        # Save generated samples
        if result.generated_samples:
            save_generated_samples(result.generated_samples, output_path)
            print(f"💾 Saved {len(result.generated_samples)} samples to {output_path}")
        else:
            print("⚠️  No samples were generated successfully")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 