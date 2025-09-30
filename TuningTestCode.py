%%capture
import os
import torch

# Installazione condizionale
if "COLAB_" in "".join(os.environ.keys()):
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
    !pip install --no-deps cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
    !pip install --no-deps unsloth
else:
    !pip install unsloth

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

# Configurazione
max_seq_length = 2048
dtype = torch.float16
model_name = "unsloth/Llama-3.2-1B-Instruct"

# Caricamento modello
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=False,
)

# Setup LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Template chat per Llama 3.1 (gestisce system messages)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Funzione per formattare le conversazioni
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# CARICAMENTO DEL DATASET DAL FILE JSON
"""
dataset = load_dataset(
    'json',
    data_files=['file1.json', 'file2.json', 'file3.json'],
    split='train'
)
"""
dataset = load_dataset(
    'json',
    data_files='il_tuo_file.json',  # Sostituisci con il percorso del tuo file JSON
    split='train'
)

# Verifica la struttura del dataset
print("Struttura del dataset:")
print(f"Numero di esempi: {len(dataset)}")
print(f"Colonne: {dataset.column_names}")
print("\nPrimo esempio:")
print(dataset[0])

# Applica la formattazione delle conversazioni
dataset = dataset.map(formatting_prompts_func, batched=True)

# Verifica la formattazione
print("\nDopo la formattazione:")
print("Primi 500 caratteri del primo esempio formattato:")
print(dataset[0]["text"][:500] + "..." if len(dataset[0]["text"]) > 500 else dataset[0]["text"])

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        save_steps=30,
    ),
)

# Training solo sulle risposte (assistant)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Verifica dei labels prima del training
print("\nVerifica dei labels (mascheramento):")
sample_idx = 0
space = tokenizer(" ", add_special_tokens=False).input_ids[0]
labels = [space if x == -100 else x for x in trainer.train_dataset[sample_idx]["labels"]]
print("Testo originale:")
print(dataset[sample_idx]["text"])
print("\nLabels (dove -100 è mascherato):")
print(tokenizer.decode(labels))

# Training
print("\nStarting training...")
trainer_stats = trainer.train()

# Test del modello dopo il training
print("\nTesting the model...")
messages = [
    {"role": "system", "content": "Sei un assistente specializzato nella gestione di Assistenza Clienti al portale eCivisWeb."},
    {"role": "user", "content": "Come devo accedere al portale eCivisWeb per utilizzare i servizi disponibili?"}
]

inputs = tokenizer.apply_chat_template(
    messages, 
    return_tensors="pt", 
    add_generation_prompt=True
)
inputs = inputs.to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print("Risposta del modello:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Salvataggio del modello
model.save_pretrained_gguf("eCivisWeb_assistant", tokenizer, quantization_method="q4_k_m")
print("Model saved successfully as 'eCivisWeb_assistant'!")
