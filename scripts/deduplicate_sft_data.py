import json
import argparse

def convert_json_to_new_format(input_file_path, output_file_path, second_file_path=None):
    """
    Convert JSON file from instruction/input/output/system format to the new format
    with source and conversations structure.
    
    Args:
        input_file_path: Path to the first JSON file to convert
        output_file_path: Path to save the resulting JSON file
        second_file_path: Optional path to a second JSON file in the target format to include
    """
    result_list = []
    # Set to track user messages from the first file for deduplication
    first_file_instructions = set()
    # Counter for duplicate entries
    duplicate_count = 0
    
    # Process the first JSON file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        first_data = json.load(f)
        
        # Handle both single object and array of objects
        items = first_data if isinstance(first_data, list) else [first_data]
        
        for data in items:
            # Get user value (instruction)
            user_value = data.get("instruction", "")
            
            # Add to tracking set
            first_file_instructions.add(user_value)
            
            # Create new format
            new_format = {
                "source": "",
                "conversations": [
                    {
                        "from": "user",
                        "value": user_value
                    },
                    {
                        "from": "assistant",
                        "value": data.get("output", "")
                    }
                ]
            }
            result_list.append(new_format)
    
    # Process the second file if provided
    if second_file_path:
        with open(second_file_path, 'r', encoding='utf-8') as f:
            # Load the second JSON file
            second_data = json.load(f)
            
            # Check if it's a JSON array or single object
            second_items = second_data if isinstance(second_data, list) else [second_data]
            
            for item in second_items:
                # Check if item has the expected structure
                if "conversations" in item and len(item["conversations"]) >= 1:
                    # Find user message for deduplication check
                    duplicate_found = False
                    for conv in item["conversations"]:
                        if conv.get("from") == "user":
                            user_value = conv.get("value", "")
                            # Skip if this user value exists in first file instructions
                            if user_value in first_file_instructions:
                                duplicate_found = True
                                duplicate_count += 1
                                break
                    
                    # Add item only if it's not a duplicate
                    if not duplicate_found:
                        result_list.append(item)
    
    # Write the combined result to the output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)
    
    print(f"Conversion completed. Output saved to {output_file_path}")
    print(f"Number of duplicates found and removed: {duplicate_count}")
    
    # Return stats for potential further processing
    return {
        "total_entries": len(result_list),
        "duplicate_count": duplicate_count
    }

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file1', type=str, required=True, help='Path to the first JSON file')
    parser.add_argument('--input_file2', type=str, help='Path to the second JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output JSON file')
    # Replace these paths with your actual file paths
    args = parser.parse_args()
    first_file_path = args.input_file1
    second_file_path = args.input_file2
    output_file_path = args.output_file
    
    convert_json_to_new_format(first_file_path, output_file_path, second_file_path)