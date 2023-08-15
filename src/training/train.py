#!/usr/bin/env python3

import os
import sys
import logging
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb
from trl import SFTTrainer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config import load_config
from src.utils.data_utils import preprocess_dataset, format_instruction
from src.utils.logger import get_logger
from src.evaluation.metrics import compute_metrics

logger = get_logger(__name__)

class CampusGPTTrainer:
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if self.config.get('wandb', {}).get('project'):
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb'].get('entity'),
                name=self.config['wandb'].get('name'),
                config=self.config,
                tags=self.config['wandb'].get('tags', [])
            )
    
    def setup_model_and_tokenizer(self):
        logger.info("loading model and tokenizer...")
        
        model_name = self.config['model']['base_model']
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.config['model'].get('cache_dir'),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=self.config['model'].get('cache_dir'),
            torch_dtype=getattr(torch, self.config['model']['torch_dtype']),
            device_map=self.config['model']['device_map'],
            load_in_4bit=self.config['model']['load_in_4bit'],
            trust_remote_code=True
        )
        
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("setup complete")
    
    def prepare_datasets(self):
        logger.info("loading datasets...")
        
        train_dataset = load_dataset(
            'json', 
            data_files=self.config['data']['train_file']
        )['train']
        
        eval_dataset = None
        if os.path.exists(self.config['data']['validation_file']):
            eval_dataset = load_dataset(
                'json',
                data_files=self.config['data']['validation_file']
            )['train']
        
        self.train_dataset = preprocess_dataset(
            train_dataset, 
            self.tokenizer, 
            self.config
        )
        
        if eval_dataset:
            self.eval_dataset = preprocess_dataset(
                eval_dataset,
                self.tokenizer,
                self.config
            )
        
        logger.info(f"train: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"eval: {len(self.eval_dataset)}")
    
    def setup_training_args(self) -> TrainingArguments:
        training_config = self.config['training']
        
        return TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            gradient_checkpointing=training_config['gradient_checkpointing'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            warmup_ratio=training_config['warmup_ratio'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            evaluation_strategy=training_config['evaluation_strategy'],
            eval_steps=training_config['eval_steps'],
            save_strategy=training_config['save_strategy'],
            save_steps=training_config['save_steps'],
            save_total_limit=training_config['save_total_limit'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            greater_is_better=training_config['greater_is_better'],
            logging_dir=training_config['logging_dir'],
            logging_strategy=training_config['logging_strategy'],
            logging_steps=training_config['logging_steps'],
            report_to=training_config['report_to'],
            fp16=training_config['fp16'],
            dataloader_num_workers=training_config['dataloader_num_workers'],
            remove_unused_columns=training_config['remove_unused_columns'],
            label_names=training_config['label_names']
        )
    
    def train(self):
        logger.info("starting training...")
        
        self.setup_model_and_tokenizer()
        self.prepare_datasets()
        
        training_args = self.setup_training_args()
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics if self.eval_dataset else None,
            dataset_text_field="text",
            max_seq_length=self.config['data']['max_seq_length']
        )
        
        logger.info("training...")
        train_result = trainer.train(
            resume_from_checkpoint=self.config['checkpointing'].get('resume_from_checkpoint')
        )
        
        trainer.save_model()
        trainer.save_state()
        
        logger.info(f"done. loss: {train_result.training_loss:.4f}")
        
        if self.eval_dataset:
            eval_results = trainer.evaluate()
            logger.info(f"eval: {eval_results}")
        
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        return train_result

def main():
    """Main training entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CampusGPT model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CampusGPTTrainer(args.config)
    
    # Override resume checkpoint if provided
    if args.resume:
        trainer.config['checkpointing']['resume_from_checkpoint'] = args.resume
    
    try:
        # Execute training
        results = trainer.train()
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise e
    
    finally:
        # Cleanup
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    main()