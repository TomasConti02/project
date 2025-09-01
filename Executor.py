%%capture
# Suppress output to keep the notebook clean during installation
import os

# Check if we're running in Colab or Kaggle
if "COLAB_" not in "".join(os.environ.keys()):
    # Local installation: install only the core Unsloth library
    !pip install unsloth
else:
    # Colab/Kaggle installation: install dependencies manually
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
    !pip install --no-deps cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
    !pip install --no-deps unsloth

##################################################################################################
# Import Unsloth's model loader
from unsloth import FastLanguageModel
import torch

# Set up model loading configuration
max_seq_length = 2048     # Maximum context length for prompts
dtype = torch.float16     # Precisione alta, ideale su T4
fourbit_models = [ #solo una lista di modelli già quantizzati a 4 bit
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "unsloth/Mistral-Small-Instruct-2409",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
    "unsloth/Llama-3.2-1B-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
]
##################################################################################################
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = False, 
    # token = "hf_..."  # Uncomment if your model requires authentication
)

##################################################################################################
# Apply Parameter-Efficient Fine-Tuning (PEFT) using LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None
)

##################################################################################################
# Setup chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

##################################################################################################
# NUOVA FUNZIONE PER IL FORMATO INSTRUCTION-INPUT-OUTPUT
def formatting_instruction_func(examples):
    """
    Converte il formato instruction-input-output in conversazioni chat
    """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Crea il contenuto dell'utente combinando instruction e input
        if input_text and input_text.strip():
            user_content = f"{instruction}\nInput: {input_text}"
        else:
            user_content = instruction
        
        # Crea la conversazione
        conversation = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
        
        # Applica il chat template
        text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    
    return {"text": texts}

##################################################################################################
# Caricamento dataset da Hugging Face
from datasets import load_dataset, Dataset

# Carica il dataset dal tuo repository Hugging Face
print("Caricando dataset da Hugging Face...")
dataset = load_dataset(
    path = "tomasconti/TestTuning",           # Il tuo repository
    data_files = "Command.json",             # Il file specifico
    split = "train"                          # Split di training
)

print(f"Dataset caricato con successo: {len(dataset)} esempi")

##################################################################################################
# Debug: controlliamo la struttura del dataset scaricato
print("Dataset scaricato da Hugging Face:")
print(f"Numero di esempi: {len(dataset)}")
print(f"Colonne disponibili: {dataset.column_names}")
print("\nPrimi 3 esempi del dataset:")
for i in range(min(3, len(dataset))):
    print(f"Esempio {i+1}:")
    print(f"  Instruction: {dataset[i]['instruction']}")
    print(f"  Input: '{dataset[i]['input']}'")
    print(f"  Output: {dataset[i]['output']}")
    print()

##################################################################################################
# Applica la funzione di formattazione
dataset = dataset.map(formatting_instruction_func, batched=True)

##################################################################################################
# Debug: controlliamo come appaiono i dati formattati
print("\nDataset dopo formattazione:")
print("Primo esempio formattato:")
print(dataset[0]["text"])
print("\n" + "="*80 + "\n")
print("Secondo esempio formattato:")
print(dataset[1]["text"])

##################################################################################################
# Setup training
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 120,  # Aumentato per il dataset più grande
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

##################################################################################################
# Train only on responses (assistant's answers)
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

##################################################################################################
# Debug: controlliamo come appaiono i labels
space = tokenizer(" ", add_special_tokens=False).input_ids[0]
sample_idx = 0
print("Labels per il primo esempio:")
decoded_labels = tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[sample_idx]["labels"]])
print(decoded_labels)

##################################################################################################
# Avvia il training
print("Iniziando il training...")
trainer_stats = trainer.train()

##################################################################################################
# Test del modello fine-tuned
print("\nTesting del modello fine-tuned:")

test_cases = [
    {"role": "user", "content": "Crea un file chiamato config.txt"},
    {"role": "user", "content": "Mostra tutti i file nella directory corrente"},
    {"role": "user", "content": "Rimuovi il file temporaneo.log"}
]

for i, test_case in enumerate(test_cases):
    print(f"\n--- Test {i+1} ---")
    messages = [test_case]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100,
        temperature=0.1,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Richiesta: {test_case['content']}")
    print(f"Risposta: {response.split('assistant')[-1].strip()}")

##################################################################################################
# Salvataggio del modello
print("\nSalvando il modello...")
model.save_pretrained("linux_commands_model")
tokenizer.save_pretrained("linux_commands_model")

# Salvataggio in formato GGUF (opzionale)
model.save_pretrained_gguf("linux_commands_model_gguf", tokenizer, quantization_method="q4_k_m")

print("Training completato e modello salvato!")
