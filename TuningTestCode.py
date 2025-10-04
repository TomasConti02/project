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
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
# =============================================================================
# TEST DI VALIDAZIONE SUL DATASET DI TRAINING
# =============================================================================

# Prima definiamo la funzione test_model correttamente con gestione device
def test_model(question, system_prompt=None, max_tokens=200):
    """Testa il modello con una domanda specifica"""
    if system_prompt is None:
        system_prompt = "Sei un assistente specializzato nella gestione di Assistenza Clienti al portale eCivisWeb."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # Applica il template chat
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )
    
    # Sposta gli input sullo stesso device del modello
    device = model.device
    inputs = inputs.to(device)
    
    # Generazione
    outputs = model.generate(
        inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Estrai solo la risposta dell'assistant
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        response = response.split("<|eot_id|>")[0].strip()
    
    return response

print("🎯 TEST DI VALIDAZIONE - CONFRONTO CON DATASET ORIGINALE")
print("=" * 60)

# Test cases direttamente dal dataset
validation_tests = [
    {
        "question": "Quali credenziali utilizzo per accedere al portale eCivisWeb?",
        "expected_topic": "SPID",
        "expected_context": "CIE"
    },
    {
        "question": "Cosa posso fare nel modulo 'Stato Contabile'?",
        "expected_topic": "movimenti bancari",
        "expected_context": "contabilità"
    },
    {
        "question": "Come funziona la registrazione dei figli su eCivisWeb?",
        "expected_topic": "Registrazione Figli",
        "expected_context": "genitore"
    },
    {
        "question": "Cliccando su 'Dati anagrafici' cosa posso verificare?",
        "expected_topic": "dati anagrafici",
        "expected_context": "corretti"
    },
    {
        "question": "Quali campi servono per la mensa scolastica?",
        "expected_topic": "codice badge",
        "expected_context": "tessera"
    }
]

def validate_response(question, response, expected_topic, expected_context):
    """Valuta la qualità della risposta"""
    score = 0
    feedback = []
    
    # Controllo topic principale
    if expected_topic.lower() in response.lower():
        score += 2
        feedback.append("✅ Topic corretto")
    else:
        feedback.append("❌ Topic mancante")
    
    # Controllo contesto
    if expected_context.lower() in response.lower():
        score += 1
        feedback.append("✅ Contesto appropriato")
    else:
        feedback.append("⚠️  Contesto debole")
    
    # Controllo lunghezza
    if len(response) > 20:
        score += 1
        feedback.append("✅ Risposta sufficientemente dettagliata")
    else:
        feedback.append("❌ Risposta troppo breve")
    
    # Controllo formattazione
    if "**" not in response and "DOMANDA:" not in response:
        score += 1
        feedback.append("✅ Formattazione naturale")
    else:
        feedback.append("⚠️  Formattazione dataset")
    
    return score, feedback

# Esegui i test
print("\n🔍 RISULTATI VALIDAZIONE:")
print("-" * 50)

total_score = 0
max_score = len(validation_tests) * 5

for i, test in enumerate(validation_tests, 1):
    print(f"\n{i}. TEST: {test['question']}")
    
    try:
        # Genera risposta
        response = test_model(test['question'], max_tokens=150)
        
        # Valuta
        score, feedback = validate_response(
            test['question'], 
            response, 
            test['expected_topic'],
            test['expected_context']
        )
        
        total_score += score
        
        print(f"   Risposta: {response}")
        print(f"   Punteggio: {score}/5")
        for fb in feedback:
            print(f"   - {fb}")
            
    except Exception as e:
        print(f"   ❌ Errore durante il test: {e}")
        response = "ERRORE"
        score = 0

# Calcola punteggio finale
if max_score > 0:
    final_score_percent = (total_score / max_score) * 100
else:
    final_score_percent = 0

print(f"\n" + "=" * 60)
print(f"📊 PUNTEGGIO FINALE: {final_score_percent:.1f}%")
print(f"({total_score}/{max_score} punti)")
print("=" * 60)

# Interpretazione del punteggio
if final_score_percent >= 80:
    print("🎉 ECCELLENTE - Il modello ha appreso perfettamente!")
elif final_score_percent >= 60:
    print("✅ BUONO - Il modello funziona bene")
elif final_score_percent >= 40:
    print("⚠️  SUFFICIENTE - Qualche area da migliorare")
else:
    print("❌ PROBLEMATICO - Necessita di più training")

# =============================================================================
# TEST SEMPLICI E RAPIDI
# =============================================================================

print("\n🧪 TEST RAPIDI DI BASE")
print("-" * 40)

simple_tests = [
    "Come accedo a eCivisWeb?",
    "Come pagare la mensa?",
    "Chi può disdire i pasti?"
]

for i, question in enumerate(simple_tests, 1):
    try:
        response = test_model(question, max_tokens=100)
        print(f"{i}. Q: {question}")
        print(f"   A: {response[:100]}...")
        print()
    except Exception as e:
        print(f"{i}. Q: {question}")
        print(f"   A: ❌ Errore: {e}")
        print()

# =============================================================================
# STATISTICHE TRAINING
# =============================================================================

print("\n📊 STATISTICHE TRAINING")
print("-" * 40)
print(f"✅ Training completato con successo!")
print(f"📈 Loss finale: {trainer_stats.training_loss:.4f}")
print(f"⏱️  Tempo di training: {trainer_stats.metrics['train_runtime']:.1f}s")

# Verifica che il modello sia addestrato
print(f"🔧 Modello device: {model.device}")
print(f"💡 Modello in memoria - NON salvato in locale")
