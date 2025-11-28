import pandas as pd
import pyarrow.parquet as pq
import json
import os
import glob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from tqdm import tqdm

def process_parquet_file(input_file, model, similarity_threshold=1.0, length_ratio_threshold=0.7, min_length_threshold=8000):
    """
    Process a single parquet file and return formatted data.
    
    Args:
        input_file: Path to the parquet file
        model: SentenceTransformer model for computing semantic similarity
        similarity_threshold: Maximum semantic similarity allowed between chosen and rejected
        length_ratio_threshold: Maximum length ratio allowed between chosen and rejected
        min_length_threshold: Minimum length for both chosen and rejected texts
        
    Returns:
        List of formatted data entries
    """
    # Read parquet file
    try:
        df = pq.read_table(input_file).to_pandas()
    except Exception as e:
        print(f"Error reading parquet file {input_file}: {str(e)}")
        return []
        
    # Create list for storing formatted data
    formatted_data = []
    
    # Iterate through each row
    for _, row in tqdm(df.iterrows(), desc="Processing rows", total=len(df), leave=False):
        try:
            problem = row['problem']
            is_reasoning_complete = row['is_reasoning_complete']
            generations = row['generations']
            correctness_1 = row['correctness_llama']
            correctness_2 = row['correctness_math_verify']
            
            # Use appropriate correctness list
            correctness = correctness_1 if correctness_1 is not None else correctness_2
            assert len(is_reasoning_complete) == len(generations) == len(correctness), "Lists must have the same length"
            
            # Filter condition: at least two true values in is_reasoning_complete
            if sum(is_reasoning_complete) >= 2:
                # Create tuples (generation, correctness, length)
                gen_info = [(gen, corr, len(gen)) for gen, corr in zip(generations, correctness)]
                
                # Filter entries with correctness = true
                correct_gens = [item for item in gen_info if item[1]]
                
                # If there are correct generations
                if correct_gens:
                    # Select the shortest correct generation
                    shortest_correct = min(correct_gens, key=lambda x: x[2])
                    
                    # Get all generations except the shortest correct one
                    other_gens = [item for item in gen_info if item != shortest_correct]
                    
                    # If there are other generations to compare with
                    if other_gens and len(shortest_correct[0]) > min_length_threshold:
                        try:
                            # Truncate shortest correct if too long
                            shortest_text = shortest_correct[0]
                            if len(shortest_text) > 10000:
                                shortest_text = shortest_text[:8000]
                            
                            # Dictionary to store (generation_index, similarity_score)
                            similarities = {}
                            
                            # Calculate similarity for each generation with a progress bar for large sets
                            gen_iterator = other_gens

                            if similarity_threshold < 1.0:
                                shortest_correct_embedding = model.encode(shortest_text, convert_to_tensor=True)
                                    
                                for i, (gen, _, _) in enumerate(gen_iterator):
                                    if len(gen) > 10000:
                                        gen_trunc = gen[:8000]
                                    else:
                                        gen_trunc = gen
                                        
                                    gen_embedding = model.encode(gen_trunc, convert_to_tensor=True)
                                    
                                    # Calculate cosine similarity
                                    similarity = cosine_similarity(
                                        shortest_correct_embedding.detach().cpu().numpy().reshape(1, -1),
                                        gen_embedding.detach().cpu().numpy().reshape(1, -1)
                                    )[0][0]
                                    
                                    similarities[i] = similarity
                            else:
                                # If similarity threshold is 1.0, assign a default similarity score of 0.0
                                # to all generations to skip similarity filtering
                                for i, _ in enumerate(gen_iterator):
                                    similarities[i] = 0.0
                            
                            # Find the generation with lowest similarity (if any exist)
                            if similarities:
                                if similarity_threshold == 1.0:
                                    # When similarity threshold is 1.0, select the longest generation
                                    min_similarity_idx = max(range(len(other_gens)), key=lambda i: other_gens[i][2])
                                else:
                                    min_similarity_idx = min(similarities, key=similarities.get)
                                min_similarity_score = similarities[min_similarity_idx]
                                
                                # Get the length of shortest correct and the candidate for rejection
                                shortest_len = len(shortest_correct[0])
                                candidate_len = len(other_gens[min_similarity_idx][0])
                                
                                # Calculate length ratio (shorter / longer)
                                length_ratio = shortest_len / candidate_len
                                
                                # Apply all three filtering criteria:
                                # 1. Similarity below threshold
                                # 2. Length ratio below threshold (significant difference in length)
                                # 3. Both texts exceed minimum length threshold
                                if ((similarity_threshold == 1.0 or min_similarity_score < similarity_threshold) and length_ratio < length_ratio_threshold and
                                    shortest_len > min_length_threshold and 
                                    candidate_len > min_length_threshold):
                                    
                                    # Create entry in required format
                                    entry = {
                                        "conversations": [
                                            {
                                                "from": "human",
                                                "value": problem
                                            }
                                        ],
                                        "chosen": {
                                            "from": "gpt",
                                            "value": shortest_correct[0]
                                        },
                                        "rejected": {
                                            "from": "gpt",
                                            "value": other_gens[min_similarity_idx][0]
                                        },
                                        # Add additional metadata for analysis
                                        "metadata": {
                                            "similarity_score": float(min_similarity_score),
                                            "chosen_length": shortest_len,
                                            "rejected_length": candidate_len,
                                            "length_ratio": float(length_ratio)
                                        }
                                    }
                                    formatted_data.append(entry)
                        except Exception as e:
                            # Log the error and continue with the next row
                            print(f"Error calculating similarities: {str(e)}")
                            continue
        except Exception as e:
            print(f"Error processing row: {str(e)}")
            continue
    
    return formatted_data

def process_directory(input_dir, output_file, similarity_threshold=1.0, length_ratio_threshold=0.7, min_length_threshold=8000, num_files=3):
    """
    Process all parquet files in the input directory and save as one JSON file.
    
    Args:
        input_dir: Directory containing parquet files
        output_file: Path to output JSON file
        similarity_threshold: Maximum semantic similarity allowed between chosen and rejected (lower = less similar)
        length_ratio_threshold: Maximum length ratio allowed between chosen and rejected (lower = bigger difference)
        min_length_threshold: Minimum length for both chosen and rejected texts
    """
    # Find all parquet files in the directory
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return
    
    if similarity_threshold == 1.0:
        model = None
        print("Skip similarity calculation")
    else:
        # Load sentence transformer model
        print("Loading sentence transformer model...")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  
        
        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Using device: {device}")

    
    process_files = parquet_files[:num_files]
    total_files = len(process_files)
    print(f"Found {total_files} parquet files to process")
    
    # Add overall progress bar for all batches
    pbar_overall = tqdm(total=total_files, desc="Overall progress", position=0)
    
    # Accumulate results from all files
    all_formatted_data = []

    # Process each parquet file in the batch
    for file_path in tqdm(process_files, desc="Files in batch", leave=False):
        print(f"Processing file: {file_path}")
        try:
            file_data = process_parquet_file(file_path, model, 
                                            similarity_threshold=similarity_threshold,
                                            length_ratio_threshold=length_ratio_threshold, 
                                            min_length_threshold=min_length_threshold)
            all_formatted_data.extend(file_data)
            print(f"  - Added {len(file_data)} DPO pairs")
            pbar_overall.update(1)  # Update the overall progress bar
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            pbar_overall.update(1)  # Update the progress bar even for failed files
            continue
    
    # Save final combined results as JSON
    print(f"Saving final results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_formatted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed and saved {len(all_formatted_data)} total DPO pairs to {output_file}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process parquet files to create DPO pairs based on multiple criteria')
    parser.add_argument('--input_dir', type=str, 
                        default="/path/to/input",
                        help='Directory containing parquet files')
    parser.add_argument('--output_file', type=str, 
                        default="/path/to/output",
                        help='Path to output JSON file')
    parser.add_argument('--similarity_threshold', type=float, default=1.0,
                        help='Maximum semantic similarity allowed (lower = less similar texts preferred)')
    parser.add_argument('--length_ratio_threshold', type=float, default=0.7,
                        help='Maximum length ratio allowed (lower = bigger length difference required)')
    parser.add_argument('--min_length_threshold', type=int, default=8000,
                        help='Minimum text length for both chosen and rejected')
    parser.add_argument('--num_files', type=int, default=3,
                        help='Number of files to process') 
       
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f"Starting processing with parameters:")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  Length ratio threshold: {args.length_ratio_threshold}")
    print(f"  Minimum length threshold: {args.min_length_threshold}")
    
    process_directory(
        args.input_dir,
        args.output_file,
        similarity_threshold=args.similarity_threshold,
        length_ratio_threshold=args.length_ratio_threshold,
        min_length_threshold=args.min_length_threshold,
        num_files=args.num_files
    )