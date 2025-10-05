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
"""
Script Completo: Training + Validazione Immediata
Esegue il fine-tuning e testa subito il modello senza salvare
"""

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
import json
from datetime import datetime
from typing import List, Dict
import re

print("🚀 TRAINING + VALIDAZIONE AUTOMATICA")
print("="*70)

# =============================================================================
# CONFIGURAZIONE
# =============================================================================
with open("optimized_config_simple.json", "r") as f:
    OPTIMIZED_CONFIG = json.load(f)

OPTIMIZED_CONFIG["dtype"] = torch.float16

FINAL_TRAINING_CONFIG = {
    **OPTIMIZED_CONFIG,
    "max_steps": 300,
    "warmup_steps": 40,
    "logging_steps": 20,
    "load_in_4bit": False,
    "lr_scheduler_type": "cosine",
}

# =============================================================================
# TEST QUESTIONS PER VALIDAZIONE
# =============================================================================
TEST_QUESTIONS = [
    {
        "id": 1,
        "category": "Accesso",
        "question": "Come posso recuperare la mia password?",
        "expected_keywords": ["area personale", "alto a destra", "dati anagrafici", "password", "login"],
        "wrong_keywords": ["dolore", "richieste", "download", "scaricare password"],
        "difficulty": "facile"
    },
    {
        "id": 2,
        "category": "Documentazione",
        "question": "Dove trovo il manuale d'uso?",
        "expected_keywords": ["documenti", "sezione", "portale", "manuali", "informative"],
        "wrong_keywords": ["moduli online", "note", "prazzi"],
        "difficulty": "facile"
    },
    {
        "id": 3,
        "category": "Accesso",
        "question": "Non riesco ad accedere al mio account",
        "expected_keywords": ["SPID", "CIE", "autenticazione", "credenziali", "comune.ecivis.it"],
        "wrong_keywords": ["database", "mezi", "servizi", "attesa"],
        "difficulty": "media"
    },
    {
        "id": 4,
        "category": "Refezione",
        "question": "Come faccio a ricaricare il conto della mensa?",
        "expected_keywords": ["pagamenti", "refezione", "ricarica", "prepagato", "saldo"],
        "wrong_keywords": ["emissione", "retta", "post-pagato"],
        "difficulty": "media"
    },
    {
        "id": 5,
        "category": "Prenotazioni",
        "question": "Come posso disdire il pasto di mio figlio?",
        "expected_keywords": ["prenotazioni", "calendario", "assenza", "disdire", "rosso"],
        "wrong_keywords": ["pagamenti", "ricarica", "emissione"],
        "difficulty": "media"
    },
    {
        "id": 6,
        "category": "Utenti",
        "question": "Dove vedo i servizi a cui è iscritto mio figlio?",
        "expected_keywords": ["utenti", "servizi", "iscritto", "refezione", "trasporto"],
        "wrong_keywords": ["pagamenti", "prenotazioni"],
        "difficulty": "facile"
    },
    {
        "id": 7,
        "category": "Generale",
        "question": "Quali sezioni posso vedere senza fare il login?",
        "expected_keywords": ["notizie", "documenti", "consultabili", "login"],
        "wrong_keywords": ["utenti", "pagamenti", "prenotazioni"],
        "difficulty": "facile"
    },
    {
        "id": 8,
        "category": "Refezione",
        "question": "Cosa significa il colore verde nel calendario delle prenotazioni?",
        "expected_keywords": ["pasto base", "presente", "verde", "mensa"],
        "wrong_keywords": ["assenza", "rosso", "pasto bianco", "giallo"],
        "difficulty": "facile"
    },
    {
        "id": 9,
        "category": "Pagamenti",
        "question": "Come funziona il servizio in post-pagato?",
        "expected_keywords": ["emissione", "retta", "post-pagato", "quota fissa"],
        "wrong_keywords": ["ricarica", "prepagato", "conto elettronico"],
        "difficulty": "difficile"
    },
    {
        "id": 10,
        "category": "Moduli",
        "question": "Posso modificare una domanda già inviata?",
        "expected_keywords": ["moduli online", "eliminare", "cancellare", "non è possibile", "modificare"],
        "wrong_keywords": ["edit", "aggiornare"],
        "difficulty": "media"
    },
]

# =============================================================================
# FUNZIONI DI VALUTAZIONE
# =============================================================================
def calculate_keyword_score(response: str, expected: List[str], wrong: List[str]) -> Dict:
    response_lower = response.lower()
    correct_found = sum(1 for kw in expected if kw.lower() in response_lower)
    correct_score = (correct_found / len(expected)) * 100 if expected else 0
    wrong_found = sum(1 for kw in wrong if kw.lower() in response_lower)
    wrong_penalty = (wrong_found / max(len(wrong), 1)) * 50
    final_score = max(0, correct_score - wrong_penalty)
    
    return {
        "correct_found": correct_found,
        "correct_total": len(expected),
        "wrong_found": wrong_found,
        "correct_score": correct_score,
        "wrong_penalty": wrong_penalty,
        "final_score": final_score,
        "found_keywords": [kw for kw in expected if kw.lower() in response_lower],
        "found_wrong": [kw for kw in wrong if kw.lower() in response_lower]
    }

def evaluate_coherence(response: str) -> Dict:
    word_count = len(response.split())
    length_ok = 10 <= word_count <= 150
    incomplete = response.endswith((' ', '\n')) or len(response) < 20
    words = response.lower().split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    no_repetition = unique_ratio > 0.5
    has_strange_chars = bool(re.search(r'[^\w\s.,;:!?\-àèéìòù()\'\"]+', response))
    
    coherence_score = 0
    if length_ok: coherence_score += 30
    if not incomplete: coherence_score += 30
    if no_repetition: coherence_score += 20
    if not has_strange_chars: coherence_score += 20
    
    return {
        "word_count": word_count,
        "length_ok": length_ok,
        "complete": not incomplete,
        "no_repetition": no_repetition,
        "no_strange_chars": not has_strange_chars,
        "coherence_score": coherence_score
    }

def get_overall_rating(keyword_score: float, coherence_score: float) -> str:
    total_score = (keyword_score * 0.6) + (coherence_score * 0.4)
    if total_score >= 80: return "ECCELLENTE"
    elif total_score >= 60: return "BUONO"
    elif total_score >= 40: return "SUFFICIENTE"
    else: return "INSUFFICIENTE"

# =============================================================================
# FASE 1: TRAINING
# =============================================================================
print("\n" + "="*70)
print("FASE 1: TRAINING DEL MODELLO")
print("="*70)

print("\nCaricamento dataset...")
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
    print(f"  {data_file}: {len(ds)} esempi")

full_dataset = concatenate_datasets(loaded_datasets)
print(f"\nDataset completo: {len(full_dataset)} esempi")

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

print("\nInizializzazione modello...")
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
        output_dir=None,
        report_to="none",
        save_strategy="no",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

print("\nInizio training...")
trainer_stats = trainer.train()
print(f"\nTraining completato! Loss finale: {trainer_stats.training_loss:.4f}")

# =============================================================================
# FASE 2: VALIDAZIONE IMMEDIATA
# =============================================================================
print("\n" + "="*70)
print("FASE 2: VALIDAZIONE DEL MODELLO")
print("="*70)

FastLanguageModel.for_inference(model)

results = []
total_keyword_score = 0
total_coherence_score = 0

for i, test in enumerate(TEST_QUESTIONS, 1):
    print(f"\n[{i}/{len(TEST_QUESTIONS)}] {test['category']}: {test['question']}")
    
    messages = [
        {"role": "system", "content": "Sei un assistente specializzato nella gestione di Assistenza Clienti al portale eCivisWeb."},
        {"role": "user", "content": test['question']}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    keyword_eval = calculate_keyword_score(
        response, 
        test['expected_keywords'], 
        test['wrong_keywords']
    )
    coherence_eval = evaluate_coherence(response)
    rating = get_overall_rating(
        keyword_eval['final_score'], 
        coherence_eval['coherence_score']
    )
    
    print(f"  Risposta: {response[:100]}{'...' if len(response) > 100 else ''}")
    print(f"  Keyword: {keyword_eval['correct_found']}/{keyword_eval['correct_total']} | Coerenza: {coherence_eval['coherence_score']}/100 | Rating: {rating}")
    
    total_keyword_score += keyword_eval['final_score']
    total_coherence_score += coherence_eval['coherence_score']
    
    results.append({
        "test_id": test['id'],
        "question": test['question'],
        "response": response,
        "keyword_score": keyword_eval['final_score'],
        "coherence_score": coherence_eval['coherence_score'],
        "rating": rating,
        "details": {"keyword_eval": keyword_eval, "coherence_eval": coherence_eval}
    })

# =============================================================================
# REPORT FINALE
# =============================================================================
print("\n" + "="*70)
print("REPORT FINALE")
print("="*70)

avg_keyword = total_keyword_score / len(TEST_QUESTIONS)
avg_coherence = total_coherence_score / len(TEST_QUESTIONS)
overall_score = (avg_keyword * 0.6) + (avg_coherence * 0.4)

print(f"\nPUNTEGGI MEDI:")
print(f"  Contenuto (keyword): {avg_keyword:.1f}/100")
print(f"  Coerenza: {avg_coherence:.1f}/100")
print(f"  Punteggio complessivo: {overall_score:.1f}/100")

rating_counts = {}
for result in results:
    rating = result['rating']
    rating_counts[rating] = rating_counts.get(rating, 0) + 1

print(f"\nDISTRIBUZIONE RATING:")
for rating in ["ECCELLENTE", "BUONO", "SUFFICIENTE", "INSUFFICIENTE"]:
    count = rating_counts.get(rating, 0)
    percentage = (count / len(TEST_QUESTIONS)) * 100
    print(f"  {rating}: {count}/{len(TEST_QUESTIONS)} ({percentage:.1f}%)")

print(f"\nVALUTAZIONE FINE-TUNING:")
if overall_score >= 80:
    print("  ECCELLENTE - Il fine-tuning è molto efficace!")
elif overall_score >= 60:
    print("  BUONO - Il fine-tuning è efficace!")
elif overall_score >= 40:
    print("  SUFFICIENTE - Il fine-tuning ha alcuni problemi.")
else:
    print("  INSUFFICIENTE - Il fine-tuning non è efficace.")

print(f"\nTRAINING STATS:")
print(f"  Loss finale: {trainer_stats.training_loss:.4f}")
print(f"  Steps completati: {FINAL_TRAINING_CONFIG['max_steps']}")
print(f"  Learning rate: {FINAL_TRAINING_CONFIG['learning_rate']:.2e}")

output_file = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "training_loss": trainer_stats.training_loss,
        "avg_keyword_score": avg_keyword,
        "avg_coherence_score": avg_coherence,
        "overall_score": overall_score,
        "rating_distribution": rating_counts,
        "detailed_results": results
    }, f, indent=2, ensure_ascii=False)

print(f"\nRisultati salvati in: {output_file}")
print("\nVALIDAZIONE COMPLETATA!")
