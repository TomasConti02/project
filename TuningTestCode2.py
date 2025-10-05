%%capture
import os
import torch

# Installazione completa
if "COLAB_" in "".join(os.environ.keys()):
    !pip install --no-deps bitsandbytes accelerate peft trl
    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
    !pip install --no-deps unsloth unsloth_zoo
else:
    !pip install unsloth unsloth_zoo

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

# =============================================================================
# CONFIGURAZIONE SEMPLICE E DIRETTA
# =============================================================================
"""
TUNING_CONFIG = {
  #60%
  #1.0862
  # Dataset originale: 84.0%
  #Generalizzazione: 22.2%
    # Configurazione modello
    "model_name": "unsloth/Llama-3.2-1B-Instruct",
    "max_seq_length": 2048,
    "dtype": torch.float16,
    "load_in_4bit": False,

    # Configurazione LoRA - MEDIA
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

    # Configurazione training - CLASSICA
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_steps": 150,
    "learning_rate": 2e-4,
    "warmup_steps": 20,

    # Altre configurazioni
    "weight_decay": 0.01,
    "logging_steps": 5,
    "save_steps": 75,
    "seed": 3407,
}
TUNING_CONFIG = {
    # 0.5426, 44%
    #Dataset originale: 100.0%
    #Generalizzazione: 44.4%
    #TRAINING PARZIALE - Il modello ricorda il dataset ma generalizza poco
    #CONSIGLIO: Training ottimale! Il modello è pronto per l'uso
    # Configurazione modello
    "model_name": "unsloth/Llama-3.2-1B-Instruct",
    "max_seq_length": 2048,
    "dtype": torch.float16,
    "load_in_4bit": False,

    # Configurazione LoRA - MEDIA per bilanciare memorizzazione e generalizzazione
    "lora_r": 12,
    "lora_alpha": 24,
    "lora_dropout": 0.15,  # Aumentato per forzare generalizzazione
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # Configurazione training - PIÙ CONSERVATIVA
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_steps": 250,       # Aumentato gradualmente
    "learning_rate": 8e-5,  # Più basso per apprendimento più profondo
    "warmup_steps": 40,     # Warmup più lungo

    # Altre configurazioni
    "weight_decay": 0.03,   # Più regolarizzazione
    "logging_steps": 10,
    "save_steps": 125,
    "seed": 3407,
}
"""
print("=== CONFIGURAZIONE SEMPLICE ===")
print("Strategia: Ritorno al basics con verifiche")
TUNING_CONFIG = {
    # %56
    #Loss finale: 1.4664
    #Dataset originale: 84.0%
    #Generalizzazione: 77.8%
    # --- CONFIGURAZIONE MODELLO ---
    "model_name": "unsloth/Llama-3.2-1B-Instruct",  # Nome del modello base da caricare (Llama 3.2 da 1 miliardo di parametri)
    "max_seq_length": 2048,                         # Lunghezza massima di input in token (finestra di contesto)
    "dtype": torch.float16,                         # Tipo di dato usato per i pesi (FP16 per velocità e memoria)
    "load_in_4bit": False,                          # Se True, carica il modello in 4 bit (quantizzazione per risparmiare RAM GPU)

    # --- CONFIGURAZIONE LoRA (Low-Rank Adaptation) ---
    "lora_r": 4,                                    # Dimensione del rank LoRA (più alto = più capacità di adattamento)
    "lora_alpha": 8,                                # Fattore di scalatura dell’adattamento LoRA
    "lora_dropout": 0.05,                           # Percentuale di dropout nei layer LoRA (per evitare overfitting)
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Layer del trasformatore su cui applicare LoRA

    # --- CONFIGURAZIONE TRAINING ---
    "batch_size": 2,                                # Numero di campioni per batch (più alto = più memoria GPU)
    "gradient_accumulation_steps": 4,               # Numero di step di accumulo prima dell’update (simula batch più grandi)
    "max_steps": 120,                               # Numero massimo di step di addestramento (definisce la durata del training)
    "learning_rate": 1.5e-4,                        # Tasso di apprendimento (quanto velocemente il modello si adatta)
    "warmup_steps": 20,                             # Step iniziali con LR crescente per stabilizzare l’addestramento

    # --- ALTRE CONFIGURAZIONI ---
    "weight_decay": 0.01,                           # Penalizzazione sui pesi per evitare overfitting
    "logging_steps": 5,                             # Ogni quanti step loggare i progressi
    "save_steps": 60,                               # Ogni quanti step salvare un checkpoint del modello
    "seed": 3407,                                   # Seed per la riproducibilità dei risultati
}

# =============================================================================
# INIZIALIZZAZIONE
# =============================================================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=TUNING_CONFIG["model_name"],
    max_seq_length=TUNING_CONFIG["max_seq_length"],
    dtype=TUNING_CONFIG["dtype"],
    load_in_4bit=TUNING_CONFIG["load_in_4bit"],
)

model = FastLanguageModel.get_peft_model(
    model,
    r=TUNING_CONFIG["lora_r"],
    target_modules=TUNING_CONFIG["lora_target_modules"],
    lora_alpha=TUNING_CONFIG["lora_alpha"],
    lora_dropout=TUNING_CONFIG["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=TUNING_CONFIG["seed"],
)

# =============================================================================
# CARICAMENTO E VERIFICA DATASET
# =============================================================================

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

dataset_esplorativo = load_dataset(
    path="tomasconti/TestTuning",
    data_files=['dataset_esplorativo.json'],
    split='train'
)

dataset_ultra_conservativo = load_dataset(
    path="tomasconti/TestTuning",
    data_files=['dataset_ultra_conservativo.json'],
    split='train'
)

dataset = concatenate_datasets([
    dataset_conservativo,
    dataset_bilanciato,
    dataset_creativo,
    dataset_esplorativo,
    dataset_ultra_conservativo
])

total_examples = len(dataset)
print(f"Dataset: {total_examples} esempi")

# VERIFICA CRITICA DEL DATASET
print("\n=== VERIFICA DATASET ===")
for i in range(min(3, len(dataset))):
    print(f"\n--- Esempio {i+1} ---")
    conv = dataset[i]['conversations']
    for msg in conv:
        print(f"{msg['role']}: {msg['content'][:100]}...")

# =============================================================================
# FORMATTAZIONE CON VERIFICA ESPLICITA
# =============================================================================

def exact_llama_template(messages):
    """Template che preserva ESATTAMENTE il contenuto"""
    formatted_text = "<|begin_of_text|>"

    for message in messages:
        role = message["role"]
        content = message["content"].strip()

        if role == "system":
            formatted_text += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "user":
            formatted_text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "assistant":
            formatted_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"

    # IMPORTANTE: Aggiungi SOLO se non c'è già assistant
    if messages[-1]["role"] != "assistant":
        formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return formatted_text

def formatting_func(examples):
    texts = []
    for conv in examples["conversations"]:
        text = exact_llama_template(conv)
        texts.append(text)
    return {"text": texts}

# Applica formattazione e RIMUOVI colonne originali
dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

# VERIFICA della formattazione
print("\n=== VERIFICA FORMATTAZIONE ===")
for i in range(min(2, len(dataset))):
    print(f"\nEsempio {i+1} formattato (primi 300 caratteri):")
    print(dataset[i]['text'][:300] + "...")

# =============================================================================
# TRAINING CON MASCHERAMENTO VERIFICATO
# =============================================================================

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=TUNING_CONFIG["max_seq_length"],
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
    ),
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=TUNING_CONFIG["batch_size"],
        gradient_accumulation_steps=TUNING_CONFIG["gradient_accumulation_steps"],
        warmup_steps=TUNING_CONFIG["warmup_steps"],
        max_steps=TUNING_CONFIG["max_steps"],
        learning_rate=TUNING_CONFIG["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=TUNING_CONFIG["logging_steps"],
        optim="adamw_torch",  # CAMBIATO: più stabile
        weight_decay=TUNING_CONFIG["weight_decay"],
        lr_scheduler_type="linear",
        seed=TUNING_CONFIG["seed"],
        output_dir="simple_tuning",
        report_to="none",
        save_steps=TUNING_CONFIG["save_steps"],
        eval_steps=None,
        save_total_limit=1,
    ),
)

# VERIFICA del mascheramento prima di applicarlo
print("\n=== VERIFICA MASCHERAMENTO ===")
test_example = dataset[0]['text']
print("Test mascheramento su primo esempio:")
print(test_example[:200] + "...")

# Applica mascheramento
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# =============================================================================
# TRAINING
# =============================================================================

print(f"\n🎯 TRAINING CON VERIFICHE")
print("=" * 50)

trainer_stats = trainer.train()
print(f"Training completato - Loss: {trainer_stats.training_loss:.4f}")
