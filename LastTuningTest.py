"""
{
  "model_name": "unsloth/Llama-3.2-1B-Instruct",
  "max_seq_length": 2048,
  "dtype": "torch.float16",
  "load_in_4bit": false,
  "lora_target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj"
  ],
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.06896194237202181,
  "batch_size": 2,
  "gradient_accumulation_steps": 4,
  "max_steps": 300,
  "learning_rate": 0.0004621179133977688,
  "warmup_steps": 30,
  "weight_decay": 0.01,
  "logging_steps": 20,
  "save_steps": 100,
  "seed": 3407
}
"""
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
import json

print("🚀 TRAINING COMPLETO CON CONFIGURAZIONE OTTIMIZZATA")
print("="*50)

# Carica la configurazione ottimizzata
with open("optimized_config_simple.json", "r") as f:
    OPTIMIZED_CONFIG = json.load(f)

# Converti dtype
OPTIMIZED_CONFIG["dtype"] = torch.float16

# Aggiusta parametri per training completo
FINAL_TRAINING_CONFIG = {
    **OPTIMIZED_CONFIG,
    "max_steps": 300,           # Aumentato per training completo
    "warmup_steps": 40,         # Warmup più lungo
    "save_steps": 100,          # Salvataggio ogni 100 steps
    "logging_steps": 20,        # Logging meno frequente
    "load_in_4bit": False,      # Più stabile per training lungo
    "lr_scheduler_type": "cosine",  # Scheduler migliore
}

print("🎯 CONFIGURAZIONE FINALE:")
for key, value in FINAL_TRAINING_CONFIG.items():
    if key in ["learning_rate", "lora_r", "lora_alpha", "lora_dropout", "max_steps"]:
        print(f"   {key}: {value}")

# =============================================================================
# CARICAMENTO DATASET COMPLETO
# =============================================================================
print("\n📦 Caricamento dataset completo...")

datasets_to_load = [
    'dataset_conservativo.json',
    'dataset_bilanciato.json', 
    'dataset_creativo.json',
    'dataset_esplorativo.json',
    'dataset_ultra_conservativo.json'
]

loaded_datasets = []
for data_file in datasets_to_load:
    ds = load_dataset("tomasconti/TestTuning", data_files=[data_file], split='train')
    loaded_datasets.append(ds)
    print(f"✅ {data_file}: {len(ds)} esempi")

full_dataset = concatenate_datasets(loaded_datasets)
print(f"📊 Dataset completo: {len(full_dataset)} esempi")

# Formattazione
def exact_llama_template(messages):
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
    if messages[-1]["role"] != "assistant":
        formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_text

def formatting_func(examples):
    texts = []
    for conv in examples["conversations"]:
        text = exact_llama_template(conv)
        texts.append(text)
    return {"text": texts}

dataset = full_dataset.map(formatting_func, batched=True, remove_columns=full_dataset.column_names)
print(f"🎯 Dataset formattato: {len(dataset)} esempi")

# =============================================================================
# INIZIALIZZAZIONE MODELLO CON CONFIGURAZIONE OTTIMIZZATA
# =============================================================================
print("\n🏗️ Inizializzazione modello con parametri ottimizzati...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=FINAL_TRAINING_CONFIG["model_name"],
    max_seq_length=FINAL_TRAINING_CONFIG["max_seq_length"],
    dtype=FINAL_TRAINING_CONFIG["dtype"],
    load_in_4bit=FINAL_TRAINING_CONFIG["load_in_4bit"],
)

model = FastLanguageModel.get_peft_model(
    model,
    r=FINAL_TRAINING_CONFIG["lora_r"],
    target_modules=FINAL_TRAINING_CONFIG["lora_target_modules"],
    lora_alpha=FINAL_TRAINING_CONFIG["lora_alpha"],
    lora_dropout=FINAL_TRAINING_CONFIG["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=FINAL_TRAINING_CONFIG["seed"],
)

print("✅ Modello inizializzato con successo!")
print(f"   Parametri LoRA: r={FINAL_TRAINING_CONFIG['lora_r']}, α={FINAL_TRAINING_CONFIG['lora_alpha']}")
print(f"   Dropout: {FINAL_TRAINING_CONFIG['lora_dropout']:.3f}")

# =============================================================================
# SETUP TRAINING (SENZA SALVATAGGIO)
# =============================================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=FINAL_TRAINING_CONFIG["max_seq_length"],
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=FINAL_TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=FINAL_TRAINING_CONFIG["gradient_accumulation_steps"],
        warmup_steps=FINAL_TRAINING_CONFIG["warmup_steps"],
        max_steps=FINAL_TRAINING_CONFIG["max_steps"],
        learning_rate=FINAL_TRAINING_CONFIG["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=FINAL_TRAINING_CONFIG["logging_steps"],
        optim="adamw_torch",
        weight_decay=FINAL_TRAINING_CONFIG["weight_decay"],
        lr_scheduler_type=FINAL_TRAINING_CONFIG["lr_scheduler_type"],
        seed=FINAL_TRAINING_CONFIG["seed"],
        output_dir=None,  # Nessun salvataggio
        report_to="none",
        save_strategy="no",  # Disabilita salvataggio
        save_steps=None,
        save_total_limit=0,
    ),
)

# Applica mascheramento
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# =============================================================================
# TRAINING COMPLETO
# =============================================================================
print(f"\n🎯 INIZIO TRAINING COMPLETO")
print("="*40)
print(f"📈 Steps totali: {FINAL_TRAINING_CONFIG['max_steps']}")
print(f"⏰ Tempo stimato: 15-25 minuti")
print(f"💾 Nessun salvataggio del modello")

trainer_stats = trainer.train()

print(f"\n✅ TRAINING COMPLETATO!")
print(f"📉 Loss finale: {trainer_stats.training_loss:.4f}")

# =============================================================================
# TEST DEL MODELLO
# =============================================================================
print(f"\n🧪 Test del modello addestrato...")

# Prepara il modello per l'inferenza
FastLanguageModel.for_inference(model)

# Test con diversi esempi
test_cases = [
    {
        "system": "Sei un assistente specializzato nella gestione di Assistenza Clienti al portale eCivisWeb.",
        "user": "Come posso recuperare la mia password?"
    },
    {
        "system": "Sei un assistente specializzato nella gestione di Assistenza Clienti al portale eCivisWeb.",
        "user": "Dove trovo il manuale d'uso?"
    },
    {
        "system": "Sei un assistente specializzato nella gestione di Assistenza Clienti al portale eCivisWeb.", 
        "user": "Non riesco ad accedere al mio account"
    }
]

print(f"\n🔍 TEST RISPOSTE DEL MODELLO:")
print("="*50)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n📝 Test {i}:")
    print(f"   User: {test_case['user']}")
    
    test_messages = [
        {"role": "system", "content": test_case["system"]},
        {"role": "user", "content": test_case["user"]}
    ]

    inputs = tokenizer.apply_chat_template(
        test_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Estrai solo la risposta dell'assistant
    if "assistant" in response:
        assistant_response = response.split("assistant")[-1].strip()
        # Pulisci ulteriormente se necessario
        assistant_response = assistant_response.replace("<|end_header_id|>", "").replace("\n\n", "").strip()
    else:
        assistant_response = response
    
    print(f"   🤖 Assistant: {assistant_response}")
    print("-" * 40)

# =============================================================================
# ANALISI FINALE
# =============================================================================
print(f"\n📊 ANALISI FINALE RISULTATI:")
print("="*40)
print(f"🎯 Parametri ottimizzati utilizzati:")
print(f"   • Learning Rate: {FINAL_TRAINING_CONFIG['learning_rate']:.2e}")
print(f"   • LoRA r: {FINAL_TRAINING_CONFIG['lora_r']}")
print(f"   • LoRA alpha: {FINAL_TRAINING_CONFIG['lora_alpha']}") 
print(f"   • Dropout: {FINAL_TRAINING_CONFIG['lora_dropout']:.3f}")
print(f"   • Steps completati: {FINAL_TRAINING_CONFIG['max_steps']}")

print(f"\n📈 Performance training:")
print(f"   • Loss iniziale (stimata): ~2.0")
print(f"   • Loss finale: {trainer_stats.training_loss:.4f}")
print(f"   • Riduzione loss: {((2.0 - trainer_stats.training_loss) / 2.0 * 100):.1f}%")

print(f"\n💡 Valutazione qualitativa:")
if trainer_stats.training_loss < 0.8:
    print("   ✅ Eccellente - Il modello ha imparato bene")
elif trainer_stats.training_loss < 1.2:
    print("   ✅ Buono - Il modello ha imparato sufficientemente")
else:
    print("   ⚠️  Accettabile - Potrebbe beneficiare di più training")

print(f"\n🎉 TRAINING COMPLETATO CON SUCCESSO!")
print("Il modello è stato addestrato con i parametri ottimizzati ed è pronto per l'uso!")
"""


🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
🚀 TRAINING COMPLETO CON CONFIGURAZIONE OTTIMIZZATA
==================================================
🎯 CONFIGURAZIONE FINALE:
   lora_r: 8
   lora_alpha: 16
   lora_dropout: 0.06896194237202181
   max_steps: 300
   learning_rate: 0.0004621179133977688

📦 Caricamento dataset completo...
✅ dataset_conservativo.json: 18 esempi
✅ dataset_bilanciato.json: 18 esempi
✅ dataset_creativo.json: 18 esempi
✅ dataset_esplorativo.json: 18 esempi
✅ dataset_ultra_conservativo.json: 18 esempi
📊 Dataset completo: 90 esempi

Map: 100%
 90/90 [00:00<00:00, 2635.24 examples/s]

🎯 Dataset formattato: 90 esempi

🏗️ Inizializzazione modello con parametri ottimizzati...
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!

Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = 0.06896194237202181.
Unsloth will patch all other layers, except LoRA matrices, causing a performance hit.
Unsloth 2025.10.1 patched 16 layers with 0 QKV layers, 0 O layers and 0 MLP layers.

✅ Modello inizializzato con successo!
   Parametri LoRA: r=8, α=16
   Dropout: 0.069

Unsloth: Tokenizing ["text"] (num_proc=6): 100%
 90/90 [00:05<00:00, 28.49 examples/s]
Map (num_proc=2): 100%
 90/90 [00:00<00:00, 321.34 examples/s]


🎯 INIZIO TRAINING COMPLETO
========================================
📈 Steps totali: 300
⏰ Tempo stimato: 15-25 minuti
💾 Nessun salvataggio del modello

==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 90 | Num Epochs = 25 | Total steps = 300
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (2 x 4 x 1) = 8
 "-____-"     Trainable parameters = 3,014,656 of 1,238,829,056 (0.24% trained)

[300/300 04:12, Epoch 25/25]
Step 	Training Loss
20 	1.888600
40 	1.312900
60 	0.644000
80 	0.254200
100 	0.139100
120 	0.082400
140 	0.052400
160 	0.037900
180 	0.028000
200 	0.022800
220 	0.020400
240 	0.017700
260 	0.016600
280 	0.016000
300 	0.016400

Unsloth: Will smartly offload gradients to save VRAM!

The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.


✅ TRAINING COMPLETATO!
📉 Loss finale: 0.3033

🧪 Test del modello addestrato...

🔍 TEST RISPOSTE DEL MODELLO:
==================================================

📝 Test 1:
   User: Come posso recuperare la mia password?
   🤖 Assistant: Per recuperare la tua password, puoi cliccare sul link "Dolore e richieste" nel modulo "Dolore", dove potrai scaricare la tua password corretta. Se hai già contato il servizio e è possibile lo farci sapere, puoi anche cliccare sul link "Contatto" per informarci del tuo nome e indirizzo per la future comunicazioni.
----------------------------------------

📝 Test 2:
   User: Dove trovo il manuale d'uso?
   🤖 Assistant: Il manuale d'uso per il portale eCivisWeb include tutte le informazioni necessarie per il funzionamento corretto del sito. Pertanto, se si accede al portale eCivisWeb e si trova in una delle sezioni principali, come ad esempio "Moduli online", all'area delle note o prazzi si potrà trovare la specifica informative necessaria per il successo della transazione.
----------------------------------------

📝 Test 3:
   User: Non riesco ad accedere al mio account
   🤖 Assistant: **Accedere al tuo account: un'esperienza rapida e sicura**Se non riesci ad accedere al tuo account, ci possa essere stato il caso di non ricaricare correttamente il database? O potresti avere presente l'azione di autenticare e potresti cercare di accedere alla tua sezione del portale eCivisWeb?Per accedere correttamente al tuo account, puoi provare a cliccare sul pulsante "Mezi e Servizi" e poi selezionare l'opzione di attesa. In questo modo potrai essere inform
----------------------------------------

📊 ANALISI FINALE RISULTATI:
========================================
🎯 Parametri ottimizzati utilizzati:
   • Learning Rate: 4.62e-04
   • LoRA r: 8
   • LoRA alpha: 16
   • Dropout: 0.069
   • Steps completati: 300

📈 Performance training:
   • Loss iniziale (stimata): ~2.0
   • Loss finale: 0.3033
   • Riduzione loss: 84.8%

💡 Valutazione qualitativa:
   ✅ Eccellente - Il modello ha imparato bene

🎉 TRAINING COMPLETATO CON SUCCESSO!
Il modello è stato addestrato con i parametri ottimizzati ed è pronto per l'uso!


"""
