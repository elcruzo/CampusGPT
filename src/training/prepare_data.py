#!/usr/bin/env python3
"""
Data preparation script for CampusGPT
Prepare and split campus Q&A dataset for training
"""

import os
import argparse
import json
from typing import List, Dict, Any
from pathlib import Path

from src.utils.data_utils import (
    load_jsonl_dataset, save_jsonl_dataset, split_dataset,
    filter_dataset, augment_dataset, create_sample_dataset
)
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


def create_campus_dataset() -> List[Dict[str, Any]]:
    """Create comprehensive campus dataset"""
    
    # Start with sample data
    dataset = create_sample_dataset()
    
    # Add more comprehensive examples
    additional_examples = [
        # Academic Category
        {
            "instruction": "What are the prerequisites for Computer Science 300?",
            "input": "",
            "output": "CS 300 requires completion of CS 200 (Data Structures), CS 220 (Computer Architecture), and MATH 340 (Discrete Mathematics) with grades of C or better. Students must also have completed at least 60 credit hours.",
            "category": "academic",
            "difficulty": "medium",
            "source": "course_catalog"
        },
        {
            "instruction": "How do I change my major?",
            "input": "I'm currently a Biology major but want to switch to Psychology",
            "output": "To change your major: 1) Meet with an advisor in Psychology to discuss requirements, 2) Complete a Change of Major form from the Registrar, 3) Get approval from your current advisor and the new department chair, 4) Submit by the deadline (typically mid-semester). For Biology to Psychology, you'll need to complete prerequisite courses like PSY 101 and STAT 200.",
            "category": "academic",
            "difficulty": "hard",
            "source": "advising_office"
        },
        {
            "instruction": "When do grades get posted?",
            "input": "",
            "output": "Final grades are typically posted within 72 hours after the final exam. Midterm grades are available by the 8th week of the semester. You can check grades on the student portal at any time once they're submitted by faculty.",
            "category": "academic",
            "difficulty": "easy",
            "source": "registrar_policies"
        },
        
        # Student Life Category
        {
            "instruction": "What meal plans are available?",
            "input": "I'm a freshman living in the dorms",
            "output": "Freshmen living in residence halls can choose from three meal plans: Unlimited (unlimited dining hall access), 14 Meals + $200 Flex (14 meals/week plus flex dollars), or 10 Meals + $300 Flex. All plans include access to the Late Night Grill. The Unlimited plan is recommended for freshmen.",
            "category": "student_life",
            "difficulty": "medium",
            "source": "dining_services"
        },
        {
            "instruction": "How do I join clubs and organizations?",
            "input": "",
            "output": "Join clubs through the Student Activities Fair in September, browse the online organization directory, or attend club meetings. Most clubs welcome new members year-round. Popular options include academic honor societies, recreational sports, volunteer organizations, and special interest groups. Contact the Student Life Office for a complete list.",
            "category": "student_life",
            "difficulty": "easy",
            "source": "student_activities"
        },
        {
            "instruction": "What are the residence hall quiet hours?",
            "input": "",
            "output": "Quiet hours are Sunday-Thursday 10 PM to 8 AM, and Friday-Saturday 12 AM to 10 AM. During finals week, 24-hour quiet hours are in effect. Courtesy hours (being respectful of noise levels) are in effect at all other times.",
            "category": "student_life",
            "difficulty": "easy",
            "source": "housing_policies"
        },
        
        # Administrative Category
        {
            "instruction": "How do I apply for financial aid?",
            "input": "I'm a continuing student",
            "output": "Continuing students must complete the FAFSA (Free Application for Federal Student Aid) annually by February 15th for priority consideration. Log into your FSA account, complete the form using tax information, and list our school code. Results are typically available 4-6 weeks after submission. Contact the Financial Aid Office if you need assistance.",
            "category": "administrative",
            "difficulty": "medium",
            "source": "financial_aid"
        },
        {
            "instruction": "What is the tuition payment deadline?",
            "input": "",
            "output": "Tuition and fees are due by 5 PM on the Friday before classes begin each semester. Payment plans are available to split costs over 4 months. Late payments incur a $50 fee and may result in class cancellation. Payment can be made online, by phone, or in person at the Bursar's Office.",
            "category": "administrative",
            "difficulty": "medium",
            "source": "bursar_office"
        },
        {
            "instruction": "How do I get an official transcript?",
            "input": "",
            "output": "Order official transcripts through the National Student Clearinghouse or the Registrar's Office. Electronic transcripts are delivered instantly ($5 fee), paper transcripts take 5-7 business days ($10 fee). You'll need to provide recipient information and verify your identity. Unofficial transcripts are free through the student portal.",
            "category": "administrative",
            "difficulty": "easy",
            "source": "registrar_office"
        },
        
        # Campus Services Category
        {
            "instruction": "What are the library hours during finals week?",
            "input": "",
            "output": "During finals week, the Main Library extends hours to 24/7 access Monday-Friday, with Saturday 7 AM - 11 PM and Sunday 10 AM - 11 PM. The Science Library is open until 2 AM Monday-Friday. Study rooms can be reserved online and quiet study areas are available on floors 3-5.",
            "category": "campus_services",
            "difficulty": "easy",
            "source": "library_services"
        },
        {
            "instruction": "How do I connect to campus WiFi?",
            "input": "My laptop won't connect to the network",
            "output": "Connect to 'CampusWiFi' network using your student ID and password. For connection issues: 1) Restart your device, 2) Forget and reconnect to the network, 3) Check for system updates, 4) Contact IT Help Desk at (555) 123-HELP or visit the Library basement for assistance. Guest WiFi is available for visitors.",
            "category": "campus_services",
            "difficulty": "easy",
            "source": "it_services"
        },
        {
            "instruction": "Where can I get health services on campus?",
            "input": "I need to see a doctor",
            "output": "The Campus Health Center is located in the Wellness Building and offers primary care, mental health services, and pharmacy services. Hours are Monday-Friday 8 AM - 5 PM. Call (555) 123-HEALTH for appointments. Emergency care is available 24/7 at the nearby University Hospital. Insurance is accepted.",
            "category": "campus_services",
            "difficulty": "medium",
            "source": "health_services"
        },
        
        # Complex/Multi-category examples
        {
            "instruction": "I'm struggling academically and need help with both my courses and financial situation",
            "input": "I'm failing two classes and might lose my financial aid",
            "output": "Contact Academic Advising immediately to discuss options like course withdrawal, incomplete grades, or tutoring resources. Visit the Learning Center for free tutoring and study skills workshops. For financial aid concerns, meet with a Financial Aid counselor to understand SAP (Satisfactory Academic Progress) requirements and potential appeal processes. Counseling services are also available for stress management.",
            "category": "academic",
            "difficulty": "hard",
            "source": "multiple_offices"
        }
    ]
    
    dataset.extend(additional_examples)
    
    logger.info(f"Created dataset with {len(dataset)} examples")
    return dataset


def prepare_dataset(input_file: str, output_dir: str, config: Dict[str, Any]) -> None:
    """
    Prepare dataset for training
    
    Args:
        input_file: Path to input dataset
        output_dir: Output directory for processed data
        config: Configuration dictionary
    """
    logger.info("Preparing CampusGPT dataset...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load or create dataset
    if input_file and Path(input_file).exists():
        logger.info(f"Loading dataset from {input_file}")
        data = load_jsonl_dataset(input_file)
    else:
        logger.info("Creating sample campus dataset")
        data = create_campus_dataset()
    
    if not data:
        raise ValueError("No data to process")
    
    # Apply data configuration
    data_config = config.get('data', {})
    
    # Filter dataset
    if data_config.get('remove_duplicates', True) or data_config.get('min_length'):
        logger.info("Filtering dataset...")
        data = filter_dataset(
            data,
            min_length=data_config.get('min_length', 10),
            max_length=data_config.get('max_length', 2000),
            remove_duplicates=data_config.get('remove_duplicates', True)
        )
    
    # Augment dataset if specified
    augmentation_factor = data_config.get('augmentation_factor', 1.0)
    if augmentation_factor > 1.0:
        logger.info(f"Augmenting dataset by factor {augmentation_factor}")
        data = augment_dataset(data, augmentation_factor)
    
    # Split dataset
    train_ratio = 1.0 - data_config.get('validation_split', 0.1) - data_config.get('test_split', 0.1)
    val_ratio = data_config.get('validation_split', 0.1)
    test_ratio = data_config.get('test_split', 0.1)
    
    train_data, val_data, test_data = split_dataset(
        data, train_ratio, val_ratio, test_ratio, 
        seed=config.get('system', {}).get('seed', 42)
    )
    
    # Save splits
    save_jsonl_dataset(train_data, output_path / "campus_train.jsonl")
    save_jsonl_dataset(val_data, output_path / "campus_val.jsonl") 
    save_jsonl_dataset(test_data, output_path / "campus_test.jsonl")
    
    # Save dataset statistics
    stats = {
        'total_examples': len(data),
        'train_examples': len(train_data),
        'val_examples': len(val_data),
        'test_examples': len(test_data),
        'categories': {},
        'difficulty_distribution': {},
        'avg_instruction_length': 0,
        'avg_output_length': 0
    }
    
    # Calculate statistics
    categories = {}
    difficulties = {}
    instruction_lengths = []
    output_lengths = []
    
    for example in data:
        category = example.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
        
        difficulty = example.get('difficulty', 'unknown')
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        instruction_lengths.append(len(example.get('instruction', '').split()))
        output_lengths.append(len(example.get('output', '').split()))
    
    stats['categories'] = categories
    stats['difficulty_distribution'] = difficulties
    stats['avg_instruction_length'] = sum(instruction_lengths) / len(instruction_lengths)
    stats['avg_output_length'] = sum(output_lengths) / len(output_lengths)
    
    # Save statistics
    with open(output_path / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Dataset preparation complete!")
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Prepare CampusGPT training data")
    parser.add_argument(
        "--input", 
        type=str,
        help="Input JSONL file (optional, will create sample data if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Training configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Prepare dataset
        prepare_dataset(args.input, args.output, config)
        
        logger.info("Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise e


if __name__ == "__main__":
    main()