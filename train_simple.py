#!/usr/bin/env python3

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            text = f"Question: {item['instruction']}\n"
            if item['input']:
                text += f"Context: {item['input']}\n"
            text += f"Answer: {item['output']}"
            data.append({'text': text, 'category': item['category']})
    return data

def main():
    print("training campusgpt with dialogpt (simple)")
    
    model_name = "microsoft/DialoGPT-small"
    
    print(f"loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    print("loading data...")
    train_data = load_data('data/raw/campus_qa.jsonl')
    
    dataset = Dataset.from_list(train_data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=256
        )
    
    print("tokenizing...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    training_args = TrainingArguments(
        output_dir="./models/campusgpt-simple",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        logging_dir='./logs',
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("training...")
    trainer.train()
    
    print("saving...")
    trainer.save_model("./models/campusgpt-simple-final")
    tokenizer.save_pretrained("./models/campusgpt-simple-final")
    
    metrics = trainer.evaluate()
    with open("./models/campusgpt-simple-final/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"done. eval loss: {metrics['eval_loss']:.4f}")
    print(f"saved to ./models/campusgpt-simple-final")
    
    test_prompt = "Question: What are the library hours?\nAnswer:"
    inputs = tokenizer.encode(test_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, temperature=0.7, pad_token_id=tokenizer.eos_token_id, do_sample=True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"test:\n{response}")

if __name__ == "__main__":
    main()