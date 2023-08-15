"""
Data processing utilities for CampusGPT
Dataset preparation, formatting, and preprocessing
"""

import json
import random
from typing import Dict, Any, List, Optional
from pathlib import Path
from datasets import Dataset
from transformers import PreTrainedTokenizer

from .logger import get_logger

logger = get_logger(__name__)


def format_instruction(
    instruction: str,
    input_text: str = "",
    output: str = "",
    template: str = "alpaca"
) -> str:
    """
    Format instruction-following examples using specified template
    
    Args:
        instruction: The instruction text
        input_text: Optional input context
        output: Expected output (for training)
        template: Template format (alpaca, chatml, vicuna)
        
    Returns:
        Formatted text string
    """
    if template == "alpaca":
        if input_text:
            text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
            
    elif template == "chatml":
        if input_text:
            text = f"""<|im_start|>system
You are a helpful campus assistant.
<|im_end|>
<|im_start|>user
{instruction}

Context: {input_text}
<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        else:
            text = f"""<|im_start|>system
You are a helpful campus assistant.
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
            
    elif template == "vicuna":
        if input_text:
            text = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {instruction}

Context: {input_text}

ASSISTANT: {output}"""
        else:
            text = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {instruction}

ASSISTANT: {output}"""
    else:
        raise ValueError(f"Unknown template: {template}")
    
    return text


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any]
) -> Dataset:
    """
    Preprocess dataset for training
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer instance
        config: Configuration dictionary
        
    Returns:
        Preprocessed dataset
    """
    data_config = config.get('data', {})
    max_length = data_config.get('max_seq_length', 512)
    template = data_config.get('template_name', 'alpaca')
    
    logger.info(f"Preprocessing dataset with {len(dataset)} examples")
    
    def format_examples(examples):
        """Format examples using instruction template"""
        texts = []
        
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples.get('input', [''] * len(examples['instruction']))[i]
            output = examples['output'][i]
            
            # Format using template
            formatted_text = format_instruction(
                instruction=instruction,
                input_text=input_text,
                output=output,
                template=template
            )
            texts.append(formatted_text)
        
        return {'text': texts}
    
    # Format examples
    formatted_dataset = dataset.map(
        format_examples,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Tokenize
    def tokenize_examples(examples):
        """Tokenize formatted examples"""
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        
        # Add labels (same as input_ids for causal LM)
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_examples,
        batched=True,
        remove_columns=['text']
    )
    
    logger.info(f"Tokenized dataset: {len(tokenized_dataset)} examples")
    return tokenized_dataset


def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSONL file
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of examples
    """
    data = []
    
    if not Path(file_path).exists():
        logger.warning(f"Dataset file not found: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    example = json.loads(line)
                    data.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON line: {line[:100]}... Error: {e}")
    
    logger.info(f"Loaded {len(data)} examples from {file_path}")
    return data


def save_jsonl_dataset(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save dataset to JSONL file
    
    Args:
        data: List of examples
        file_path: Output file path
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in data:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"Saved {len(data)} examples to {file_path}")


def split_dataset(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train/validation/test sets
    
    Args:
        data: Input dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train, val, test) datasets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle data
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    total_size = len(shuffled_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Split data
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    logger.info(f"Dataset split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def filter_dataset(
    data: List[Dict[str, Any]],
    min_length: int = 10,
    max_length: int = 2000,
    remove_duplicates: bool = True
) -> List[Dict[str, Any]]:
    """
    Filter dataset by length and remove duplicates
    
    Args:
        data: Input dataset
        min_length: Minimum text length
        max_length: Maximum text length
        remove_duplicates: Whether to remove duplicate examples
        
    Returns:
        Filtered dataset
    """
    filtered_data = []
    seen_instructions = set()
    
    for example in data:
        instruction = example.get('instruction', '')
        output = example.get('output', '')
        
        # Length filtering
        if len(instruction) < min_length or len(output) < min_length:
            continue
        if len(instruction) > max_length or len(output) > max_length:
            continue
        
        # Duplicate filtering
        if remove_duplicates:
            if instruction in seen_instructions:
                continue
            seen_instructions.add(instruction)
        
        filtered_data.append(example)
    
    logger.info(f"Filtered dataset: {len(data)} -> {len(filtered_data)} examples")
    return filtered_data


def augment_dataset(
    data: List[Dict[str, Any]],
    augmentation_factor: float = 1.5
) -> List[Dict[str, Any]]:
    """
    Augment dataset with paraphrased examples
    
    Args:
        data: Input dataset
        augmentation_factor: Factor to increase dataset size
        
    Returns:
        Augmented dataset
    """
    # Simple augmentation by rephrasing questions
    augmented_data = data.copy()
    
    rephrase_templates = [
        "Can you help me with: {instruction}",
        "I need information about: {instruction}",
        "Could you explain: {instruction}",
        "What can you tell me about: {instruction}"
    ]
    
    target_size = int(len(data) * augmentation_factor)
    additional_needed = target_size - len(data)
    
    for i in range(additional_needed):
        original_example = random.choice(data)
        template = random.choice(rephrase_templates)
        
        # Create augmented example
        augmented_example = original_example.copy()
        augmented_example['instruction'] = template.format(
            instruction=original_example['instruction']
        )
        
        augmented_data.append(augmented_example)
    
    logger.info(f"Augmented dataset: {len(data)} -> {len(augmented_data)} examples")
    return augmented_data


def create_sample_dataset() -> List[Dict[str, Any]]:
    """Create sample campus dataset for testing"""
    sample_data = [
        {
            "instruction": "What are the dining hall hours?",
            "input": "I'm a student looking for meal times",
            "output": "The dining halls are typically open Monday-Friday 7:00 AM - 9:00 PM, and weekends 8:00 AM - 8:00 PM. The Late Night Grill in the student union is open until midnight on weekdays.",
            "category": "student_life",
            "difficulty": "easy"
        },
        {
            "instruction": "How do I register for fall semester classes?",
            "input": "I'm a sophomore computer science major",
            "output": "To register for classes: 1) Meet with your academic advisor during your assigned time, 2) Clear any holds on your account, 3) Use the online course catalog to plan your schedule, 4) Register online starting at your priority registration time. CS sophomores typically register in mid-March.",
            "category": "academic",
            "difficulty": "medium"
        },
        {
            "instruction": "What financial aid options are available?",
            "input": "",
            "output": "Financial aid options include federal grants (Pell Grant), state grants, institutional scholarships, work-study programs, and federal student loans. To apply, complete the FAFSA by the priority deadline of February 15th. Visit the Financial Aid Office in the Student Services building for personalized assistance.",
            "category": "administrative",
            "difficulty": "medium"
        },
        {
            "instruction": "Where can I get help with computer problems?",
            "input": "My laptop won't connect to campus WiFi",
            "output": "For computer and WiFi issues, contact IT Support at the Help Desk in the Library basement, call (555) 123-HELP, or submit a ticket online. They provide free virus removal, software installation, and WiFi troubleshooting for all students.",
            "category": "campus_services",
            "difficulty": "easy"
        }
    ]
    
    return sample_data