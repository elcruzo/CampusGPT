#!/usr/bin/env python3

import os
import sys
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer
import warnings
warnings.filterwarnings('ignore')

def format_instruction(sample):
    """Format samples for instruction tuning"""
    text = f"""### Human: {sample['instruction']}
{f"Context: {sample['input']}" if sample['input'] else ""}

### Assistant: {sample['output']}"""
    return text

def main():
    print("starting llama 2 qlora fine-tuning")
    
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        print("warning: no HF token found")
    
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
    )
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"using gpu: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("using mps (apple silicon)")
        bnb_config = None
    else:
        device = "cpu"
        print("using cpu (slow)")
        bnb_config = None
    
    print(f"loading {model_name}...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            token=hf_token
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            token=hf_token
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
    except Exception as e:
        print(f"error loading model: {e}")
        print("check: llama 2 license accepted, HF token set, huggingface-cli login")
        return
    
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print("loading training data...")
    with open('data/raw/campus_qa.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    formatted_data = [{"text": format_instruction(sample)} for sample in data]
    dataset = Dataset.from_list(formatted_data)
    
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    training_args = TrainingArguments(
        output_dir="./models/llama2-campus-qlora",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=10,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        push_to_hub=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
    )
    
    print("training...")
    
    try:
        trainer.train()
        
        print("saving model...")
        trainer.save_model("./models/llama2-campus-qlora-final")
        tokenizer.save_pretrained("./models/llama2-campus-qlora-final")
        
        metrics = trainer.evaluate()
        with open("./models/llama2-campus-qlora-final/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"done. eval loss: {metrics['eval_loss']:.4f}")
        print(f"saved to ./models/llama2-campus-qlora-final")
        
        test_prompt = "### Human: What are the library hours?\n\n### Assistant:"
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)
        response = pipe(test_prompt)[0]['generated_text']
        print(f"test:\n{response}")
        
    except Exception as e:
        print(f"training error: {e}")

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    main()