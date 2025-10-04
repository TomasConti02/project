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
from datasets import load_dataset, concatenate_datasets
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

#----------------------------------------------------------------------------------#
# CONFIGURAZIONE LoRA MIGLIORATA
#----------------------------------------------------------------------------------#
model = FastLanguageModel.get_peft_model(
    model,
    r=6,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

#----------------------------------------------------------------------------------#
# TEMPLATE CHAT PER LLAMA 3.1
#----------------------------------------------------------------------------------#
#tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

#----------------------------------------------------------------------------------#
# CARICAMENTO DATASET CORRETTO
#----------------------------------------------------------------------------------#
# Carica i dataset separatamente con nomi UNIVOCI
dataset_conservativo = load_dataset(
    path="tomasconti/TestTuning",
    data_files=['dataset_conservativo.json'],
    split='train'
)

dataset_bilanciato = load_dataset(
    path="tomasconti/TestTuning", 
    data_files=['dataset_bilanciato.json'],
    split='train'
)

dataset_creativo = load_dataset(
    path="tomasconti/TestTuning",
    data_files=['dataset_creativo.json'],
    split='train'
)

# CORREZIONE: Usa nomi diversi per gli altri dataset
dataset_esplorativo = load_dataset(  # NOME DIVERSO
    path="tomasconti/TestTuning",
    data_files=['dataset_esplorativo.json'],
    split='train'
)

dataset_ultra_conservativo = load_dataset(  # NOME DIVERSO  
    path="tomasconti/TestTuning",
    data_files=['dataset_ultra_conservativo.json'],
    split='train'
)

# Unisci TUTTI i dataset
dataset = concatenate_datasets([
    dataset_conservativo, 
    dataset_bilanciato, 
    dataset_creativo,
    dataset_esplorativo,
    dataset_ultra_conservativo
])
# APPLICA IL TEMPLATE PERSONALIZZATO PER EVITARE PROBLEMI DI DATA LEAKAGE
def custom_llama_template(messages):
    """
    Template personalizzato per Llama 3.1 che non aggiunge informazioni extra
    """
    formatted_text = "<|begin_of_text|>"
    for message in messages:
        role = message["role"]
        content = message["content"].strip()  # Aggiunto strip per pulizia

        if role == "system":
            formatted_text += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "user":
            formatted_text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "assistant":
            formatted_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
    
    # AGGIUNGI TOKEN DI INIZIO RISPOSTA ASSISTENTE PER IL TRAINING
    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return formatted_text

# Funzione di formattazione corretta
def correct_formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [custom_llama_template(convo) for convo in convos]
    return {"text": texts}

# Riapplica la formattazione corretta
dataset = dataset.map(correct_formatting_prompts_func, batched=True)

# VERIFICA DEL DATASET CORRETTO
print("=== VERIFICA DATASET FINALE ===")
print(f"Numero di esempi: {len(dataset)}")
print(f"Colonne: {dataset.column_names}")

# Mostra un esempio completo per verifica
print("\n--- Esempio formattato COMPLETO ---")
print(dataset[0]['text'])
print("-" * 50)

# TOKENIZZAZIONE FINALE PER IL TRAINING
def tokenize_function(examples):
    # Tokenizza senza aggiungere token speciali (già presenti nel testo)
    tokenized = tokenizer(
        examples["text"],
        add_special_tokens=False,  # CRUCIALE: i token speciali sono già nel testo
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        return_tensors=None
    )
    
    # Aggiungi labels per il training (causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Applica la tokenizzazione finale
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizzando il dataset"
)

# VERIFICA FINALE TOKENIZZAZIONE
print("\n=== VERIFICA TOKENIZZAZIONE ===")
example_tokens = tokenized_dataset[0]['input_ids']
example_decoded = tokenizer.decode(example_tokens, skip_special_tokens=False)
example_original = dataset[0]['text']

# Verifica più robusta
is_correct = example_original in example_decoded
print(f"Encoding/decoding preserva il testo: {is_correct}")
print(f"Numero di token nell'esempio: {len(example_tokens)}")

if not is_correct:
    print("⚠ ATTENZIONE: Problemi di encoding rilevati!")
    print(f"Originale (primi 200):  {example_original[:200]}...")
    print(f"Decodificato (primi 200): {example_decoded[:200]}...")
    
    # Verifica i token speciali
    special_tokens = tokenizer.all_special_tokens
    print(f"\nToken speciali riconosciuti: {special_tokens}")
else:
    print("✓ Tokenizzazione corretta!")

# STATISTICHE DEL DATASET
token_lengths = [len(ex['input_ids']) for ex in tokenized_dataset]
print(f"\n=== STATISTICHE DATASET ===")
print(f"Lunghezza minima token: {min(token_lengths)}")
print(f"Lunghezza massima token: {max(token_lengths)}")
print(f"Lunghezza media token: {sum(token_lengths) / len(token_lengths):.1f}")

# Distribuzione delle lunghezze
import numpy as np
print(f"Lunghezza mediana token: {np.median(token_lengths):.1f}")
print(f"Esempi sotto i 100 token: {sum(1 for x in token_lengths if x < 100)}")
print(f"Esempi sopra i {max_seq_length} token: {sum(1 for x in token_lengths if x >= max_seq_length)}")

# VERIFICA LABELS
print(f"\n=== VERIFICA LABELS ===")
print(f"Esempio labels: {tokenized_dataset[0]['labels'][:20]}...")
print(f"Input_ids e labels sono identici: {tokenized_dataset[0]['input_ids'] == tokenized_dataset[0]['labels']}")
# Trainer MIGLIORATO
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,  # Usa dataset NON tokenizzato (SFTTrainer gestisce la tokenizzazione)
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,  # Migliora performance GPU
        padding=True,
    ),
    dataset_num_proc=2,
    packing=False,  # Giustamente False per chat templates
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=15,
        max_steps=120,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        save_steps=60,
        eval_steps=None,  # Disabilita evaluation se non hai validation set
        save_total_limit=1,  # Evita di riempire disco
    ),
)

# TRAINING SOLO SULLE RISPOSTE - CONFIGURAZIONE PERFETTA!
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
