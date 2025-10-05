import os
import torch
import json
import numpy as np
from datetime import datetime

print("🚀 OTTIMIZZAZIONE PARAMETRI CORRETTA")
print("="*50)

# Installazione pacchetti
print("📦 Verifica installazione...")
!pip install -q optuna

import optuna
from optuna.trial import TrialState

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

# =============================================================================
# CONFIGURAZIONE BASE CORRETTA
# =============================================================================
BASE_CONFIG = {
    "model_name": "unsloth/Llama-3.2-1B-Instruct",
    "max_seq_length": 2048,
    "dtype": torch.float16,
    "load_in_4bit": True,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "logging_steps": 5,
    "save_steps": 25,
    "seed": 3407,
    "max_steps": 60,
}

# =============================================================================
# CARICAMENTO DATASET SEMPLIFICATO
# =============================================================================
def load_and_prepare_data():
    """Carica dataset senza validation split complesso"""
    print("📦 Caricamento dataset...")
    
    datasets_to_load = [
        'dataset_conservativo.json',
        'dataset_bilanciato.json', 
        'dataset_creativo.json',
        'dataset_esplorativo.json',
        'dataset_ultra_conservativo.json'
    ]
    
    loaded_datasets = []
    for data_file in datasets_to_load:
        try:
            ds = load_dataset("tomasconti/TestTuning", data_files=[data_file], split='train')
            loaded_datasets.append(ds)
            print(f"✅ {data_file}: {len(ds)} esempi")
        except Exception as e:
            print(f"⚠️  Errore {data_file}: {e}")
    
    if not loaded_datasets:
        raise Exception("❌ Nessun dataset caricato")
    
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
    
    # Usa subset per tuning veloce
    if len(dataset) > 80:
        dataset = dataset.select(range(80))
    
    print(f"🎯 Dataset per tuning: {len(dataset)} esempi")
    
    return dataset

# =============================================================================
# COSTRUZIONE TRAINER SEMPLIFICATA
# =============================================================================
def build_simple_trainer(config, dataset):
    """Costruisce trainer senza validation complesso"""
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model_name"],
            max_seq_length=config["max_seq_length"],
            dtype=config["dtype"],
            load_in_4bit=config["load_in_4bit"],
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config["lora_r"],
            target_modules=config["lora_target_modules"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config["seed"],
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config["max_seq_length"],
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=config["batch_size"],
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                warmup_steps=config["warmup_steps"],
                max_steps=config["max_steps"],
                learning_rate=config["learning_rate"],
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=config["logging_steps"],
                optim="adamw_torch",
                weight_decay=config["weight_decay"],
                lr_scheduler_type="linear",
                seed=config["seed"],
                output_dir=f"trial_{config['lora_r']}_{config['learning_rate']:.1e}",
                report_to="none",
                save_steps=config["save_steps"],
                save_total_limit=1,
                # Rimossi parametri di evaluation problematici
            ),
        )

        # Applica mascheramento
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

        return trainer
        
    except Exception as e:
        print(f"❌ Errore costruzione trainer: {e}")
        raise

# =============================================================================
# FUNZIONE OBIETTIVO SEMPLIFICATA
# =============================================================================
def simple_objective(trial):
    """Funzione obiettivo semplificata ma efficace"""
    
    # Parametri con range ragionevoli
    config = BASE_CONFIG.copy()
    config["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    config["lora_r"] = trial.suggest_categorical("lora_r", [4, 8, 16, 32])
    config["lora_alpha"] = config["lora_r"] * 2  # Regola semplice
    config["lora_dropout"] = trial.suggest_float("lora_dropout", 0.01, 0.2)
    config["max_steps"] = trial.suggest_int("max_steps", 40, 80)
    
    # Log
    print(f"\n🧪 Trial {trial.number}")
    print(f"   LR: {config['learning_rate']:.2e}, LoRA r: {config['lora_r']}")
    print(f"   Dropout: {config['lora_dropout']:.3f}, Steps: {config['max_steps']}")
    
    try:
        # Costruisci e addestra
        trainer = build_simple_trainer(config, dataset)
        
        print(f"   🏃 Training in corso...")
        training_result = trainer.train()
        loss = training_result.training_loss
        
        print(f"   ✅ Loss: {loss:.4f}")
        
        # Salva informazioni
        trial.set_user_attr("config", config)
        
        # Pulisci memoria
        del trainer
        torch.cuda.empty_cache()
        
        return loss
        
    except Exception as e:
        print(f"   ❌ Trial fallito: {e}")
        torch.cuda.empty_cache()
        return float('inf')

# =============================================================================
# CALLBACK SEMPLIFICATO
# =============================================================================
class SimpleTrialCallback:
    def __init__(self, total_trials):
        self.total_trials = total_trials
        self.start_time = datetime.now()
        
    def __call__(self, study, trial):
        if trial.state == TrialState.COMPLETE:
            current_time = datetime.now()
            elapsed = (current_time - self.start_time).total_seconds() / 60
            
            completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
            progress = f"{completed}/{self.total_trials}"
            
            if completed > 0:
                avg_time = elapsed / completed
                remaining = self.total_trials - completed
                eta = avg_time * remaining
                progress += f" | ETA: {eta:.1f}min"
            
            best_loss = study.best_value if study.best_value != float('inf') else "N/A"
            print(f"📊 Progresso: {progress} | Tempo: {elapsed:.1f}min | Miglior Loss: {best_loss}")

# =============================================================================
# ANALISI RISULTATI
# =============================================================================
def analyze_results_simple(study):
    """Analisi risultati semplificata ma efficace"""
    print("\n" + "="*60)
    print("📊 RISULTATI OTTIMIZZAZIONE")
    print("="*60)
    
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    
    if not completed_trials:
        print("❌ Nessun trial completato con successo")
        return None
    
    # Statistiche
    losses = [t.value for t in completed_trials]
    best_trial = study.best_trial
    
    print(f"📈 Trial completati: {len(completed_trials)}/{len(study.trials)}")
    print(f"📉 Loss media: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"🎯 Miglior loss: {study.best_value:.4f}")
    
    # Analisi per parametro
    print(f"\n🔍 ANALISI PARAMETRI:")
    
    # Per Learning Rate
    lr_performance = {}
    for trial in completed_trials:
        lr = trial.params['learning_rate']
        lr_key = f"{lr:.1e}"
        if lr_key not in lr_performance:
            lr_performance[lr_key] = []
        lr_performance[lr_key].append(trial.value)
    
    print("   Learning Rate:")
    for lr, loss_list in sorted(lr_performance.items()):
        print(f"     {lr}: {np.mean(loss_list):.4f} (n={len(loss_list)})")
    
    # Per LoRA r
    lora_performance = {}
    for trial in completed_trials:
        lora_r = trial.params['lora_r']
        if lora_r not in lora_performance:
            lora_performance[lora_r] = []
        lora_performance[lora_r].append(trial.value)
    
    print("   LoRA r:")
    for lora_r, loss_list in sorted(lora_performance.items()):
        print(f"     {lora_r}: {np.mean(loss_list):.4f} (n={len(loss_list)})")
    
    # Top 3 configurazioni
    sorted_trials = sorted(completed_trials, key=lambda x: x.value)
    print(f"\n🏅 TOP 3 CONFIGURAZIONI:")
    for i, trial in enumerate(sorted_trials[:3]):
        print(f"\n#{i+1} - Loss: {trial.value:.4f}")
        print(f"   LR: {trial.params['learning_rate']:.2e}")
        print(f"   LoRA r: {trial.params['lora_r']}, α: {trial.params['lora_r'] * 2}")
        print(f"   Dropout: {trial.params['lora_dropout']:.3f}")
        print(f"   Steps: {trial.params['max_steps']}")
    
    return best_trial

# =============================================================================
# CONFIGURAZIONE FINALE
# =============================================================================
def create_final_config_simple(best_trial):
    """Crea configurazione finale"""
    best_params = best_trial.params
    
    FINAL_CONFIG = {
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "max_seq_length": 2048,
        "dtype": "torch.float16",
        "load_in_4bit": False,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        "lora_r": best_params["lora_r"],
        "lora_alpha": best_params["lora_r"] * 2,
        "lora_dropout": best_params["lora_dropout"],
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_steps": 300,
        "learning_rate": best_params["learning_rate"],
        "warmup_steps": 30,
        "weight_decay": 0.01,
        "logging_steps": 20,
        "save_steps": 100,
        "seed": 3407,
    }
    
    # Salva configurazione
    with open("optimized_config_simple.json", "w") as f:
        json.dump(FINAL_CONFIG, f, indent=2)
    
    return FINAL_CONFIG

# =============================================================================
# ESECUZIONE PRINCIPALE
# =============================================================================
def main_simple():
    """Esegui ottimizzazione semplificata"""
    global dataset
    
    # Configurazione
    N_TRIALS = 12
    
    try:
        # Carica dati
        dataset = load_and_prepare_data()
        
        # Configura studio Optuna
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=3407),
        )
        
        # Callback
        progress_callback = SimpleTrialCallback(N_TRIALS)
        
        # Esegui ottimizzazione
        print(f"\n🎯 INIZIO OTTIMIZZAZIONE CON {N_TRIALS} TRIAL")
        print("⏰ Tempo stimato: 30-40 minuti")
        
        study.optimize(simple_objective, n_trials=N_TRIALS, callbacks=[progress_callback])
        
        # Analisi risultati
        best_trial = analyze_results_simple(study)
        
        if best_trial:
            # Configurazione finale
            final_config = create_final_config_simple(best_trial)
            
            print(f"\n🚀 CONFIGURAZIONE FINALE OTTIMIZZATA:")
            print("="*40)
            for key, value in final_config.items():
                if key in ["learning_rate", "lora_r", "lora_alpha", "lora_dropout", "max_steps"]:
                    print(f"   {key}: {value}")
            
            print(f"\n💾 Configurazione salvata in 'optimized_config_simple.json'")
            print(f"✅ OTTIMIZZAZIONE COMPLETATA!")
            
            return final_config, study
            
    except Exception as e:
        print(f"❌ Errore durante l'ottimizzazione: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =============================================================================
# AVVIA OTTIMIZZAZIONE
# =============================================================================
if __name__ == "__main__":
    print("🚀 OTTIMIZZAZIONE PARAMETRI - VERSIONE CORRETTA")
    print("="*50)
    
    final_config, study = main_simple()
    
    if final_config:
        print(f"\n🎉 SUCCESSO! Configurazione ottimale trovata:")
        print(f"   Learning Rate: {final_config['learning_rate']:.2e}")
        print(f"   LoRA r: {final_config['lora_r']}")
        print(f"   Dropout: {final_config['lora_dropout']:.3f}")
        print(f"\n📝 Usa questi parametri per il training completo!")
    else:
        print(f"\n❌ Ottimizzazione fallita")
  """
  

🚀 OTTIMIZZAZIONE PARAMETRI CORRETTA
==================================================
📦 Verifica installazione...
🚀 OTTIMIZZAZIONE PARAMETRI - VERSIONE CORRETTA
==================================================
📦 Caricamento dataset...
✅ dataset_conservativo.json: 18 esempi
✅ dataset_bilanciato.json: 18 esempi
✅ dataset_creativo.json: 18 esempi
✅ dataset_esplorativo.json: 18 esempi
✅ dataset_ultra_conservativo.json: 18 esempi
📊 Dataset completo: 90 esempi

Map: 100%
 90/90 [00:00<00:00, 3774.72 examples/s]

🎯 Dataset per tuning: 80 esempi

🎯 INIZIO OTTIMIZZAZIONE CON 12 TRIAL
⏰ Tempo stimato: 30-40 minuti

🧪 Trial 0
   LR: 1.14e-04, LoRA r: 16
   Dropout: 0.172, Steps: 68
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!

Unsloth: Tokenizing ["text"] (num_proc=6): 100%
 80/80 [00:05<00:00, 20.44 examples/s]
Map (num_proc=2): 100%
 80/80 [00:00<00:00, 195.56 examples/s]

   🏃 Training in corso...

[68/68 01:30, Epoch 6/7]
Step 	Training Loss
5 	2.068700
10 	1.981700
15 	1.771300
20 	1.720700
25 	1.537900
30 	1.397100
35 	1.266100
40 	1.250800
45 	0.970700
50 	1.239100
55 	0.981100
60 	0.972700
65 	0.892000

   ✅ Loss: 1.3708
📊 Progresso: 1/12 | ETA: 22.0min | Tempo: 2.0min | Miglior Loss: 1.3707703141605152

🧪 Trial 1
   LR: 4.62e-04, LoRA r: 8
   Dropout: 0.069, Steps: 55
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
   🏃 Training in corso...

[55/55 01:07, Epoch 5/6]
Step 	Training Loss
5 	2.058000
10 	1.895700
15 	1.519100
20 	1.462100
25 	1.160900
30 	0.984000
35 	0.735500
40 	0.745500
45 	0.429600
50 	0.618600
55 	0.408700

   ✅ Loss: 1.0925
📊 Progresso: 2/12 | ETA: 17.0min | Tempo: 3.4min | Miglior Loss: 1.0925339525396174

🧪 Trial 2
   LR: 1.21e-04, LoRA r: 32
   Dropout: 0.078, Steps: 61
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
   🏃 Training in corso...

[61/61 01:14, Epoch 6/7]
Step 	Training Loss
5 	2.059200
10 	1.918400
15 	1.621600
20 	1.549000
25 	1.333600
30 	1.174400
35 	0.980800
40 	0.988400
45 	0.682600
50 	0.902900
55 	0.661100
60 	0.666300

   ✅ Loss: 1.2052
📊 Progresso: 3/12 | ETA: 14.7min | Tempo: 4.9min | Miglior Loss: 1.0925339525396174

🧪 Trial 3
   LR: 5.22e-05, LoRA r: 32
   Dropout: 0.110, Steps: 72
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
   🏃 Training in corso...

[72/72 01:27, Epoch 7/8]
Step 	Training Loss
5 	2.069300
10 	1.990300
15 	1.802300
20 	1.779700
25 	1.629100
30 	1.506300
35 	1.398600
40 	1.373500
45 	1.108900
50 	1.420300
55 	1.166600
60 	1.148400
65 	1.076900
70 	1.155500

   ✅ Loss: 1.4697
📊 Progresso: 4/12 | ETA: 13.3min | Tempo: 6.6min | Miglior Loss: 1.0925339525396174

🧪 Trial 4
   LR: 7.20e-05, LoRA r: 32
   Dropout: 0.058, Steps: 67
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
   🏃 Training in corso...

[67/67 01:22, Epoch 6/7]
Step 	Training Loss
5 	2.066100
10 	1.965300
15 	1.743900
20 	1.695200
25 	1.517400
30 	1.377700
35 	1.249600
40 	1.237700
45 	0.961400
50 	1.232900
55 	0.979800
60 	0.973100
65 	0.897200

   ✅ Loss: 1.3646
📊 Progresso: 5/12 | ETA: 11.6min | Tempo: 8.3min | Miglior Loss: 1.0925339525396174

🧪 Trial 5
   LR: 7.14e-05, LoRA r: 4
   Dropout: 0.049, Steps: 41
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
   🏃 Training in corso...

[41/41 00:49, Epoch 4/5]
Step 	Training Loss
5 	2.074800
10 	2.055100
15 	1.962600
20 	1.994900
25 	1.910300
30 	1.888700
35 	1.845000
40 	1.823700

   ✅ Loss: 1.9389
📊 Progresso: 6/12 | ETA: 9.4min | Tempo: 9.4min | Miglior Loss: 1.0925339525396174

🧪 Trial 6
   LR: 1.49e-04, LoRA r: 32
   Dropout: 0.162, Steps: 57
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
   🏃 Training in corso...

[57/57 01:10, Epoch 5/6]
Step 	Training Loss
5 	2.055700
10 	1.899100
15 	1.566300
20 	1.497900
25 	1.260400
30 	1.092000
35 	0.880700
40 	0.892800
45 	0.585300
50 	0.796300
55 	0.570700

   ✅ Loss: 1.1688
📊 Progresso: 7/12 | ETA: 7.7min | Tempo: 10.8min | Miglior Loss: 1.0925339525396174

🧪 Trial 7
   LR: 3.25e-04, LoRA r: 4
   Dropout: 0.075, Steps: 72
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
   🏃 Training in corso...

[72/72 01:26, Epoch 7/8]
Step 	Training Loss
5 	2.070100
10 	1.986500
15 	1.753100
20 	1.675800
25 	1.460400
30 	1.312100
35 	1.130300
40 	1.111000
45 	0.794400
50 	1.028000
55 	0.744500
60 	0.739200
65 	0.626400
70 	0.683600

   ✅ Loss: 1.2107
📊 Progresso: 8/12 | ETA: 6.3min | Tempo: 12.6min | Miglior Loss: 1.0925339525396174

🧪 Trial 8
   LR: 1.28e-04, LoRA r: 4
   Dropout: 0.138, Steps: 72
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
   🏃 Training in corso...

[72/72 01:26, Epoch 7/8]
Step 	Training Loss
5 	2.073900
10 	2.040200
15 	1.908600
20 	1.905300
25 	1.781600
30 	1.698300
35 	1.589700
40 	1.541400
45 	1.279100
50 	1.619700
55 	1.367300
60 	1.332200
65 	1.276300
70 	1.359500

   ✅ Loss: 1.6251
📊 Progresso: 9/12 | ETA: 4.8min | Tempo: 14.3min | Miglior Loss: 1.0925339525396174

🧪 Trial 9
   LR: 2.46e-04, LoRA r: 4
   Dropout: 0.058, Steps: 55
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
   🏃 Training in corso...

[55/55 01:07, Epoch 5/6]
Step 	Training Loss
5 	2.071700
10 	2.007100
15 	1.809500
20 	1.758100
25 	1.568500
30 	1.436400
35 	1.310200
40 	1.291000
45 	1.027500
50 	1.325100
55 	1.094100

   ✅ Loss: 1.5181
📊 Progresso: 10/12 | ETA: 3.1min | Tempo: 15.7min | Miglior Loss: 1.0925339525396174

🧪 Trial 10
   LR: 4.88e-04, LoRA r: 8
   Dropout: 0.012, Steps: 47
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
   🏃 Training in corso...

[47/47 00:57, Epoch 4/5]
Step 	Training Loss
5 	2.056800
10 	1.887200
15 	1.499800
20 	1.444000
25 	1.138400
30 	0.965400
35 	0.732100
40 	0.746000
45 	0.466100

   ✅ Loss: 1.1933
📊 Progresso: 11/12 | ETA: 1.5min | Tempo: 16.9min | Miglior Loss: 1.0925339525396174

🧪 Trial 11
   LR: 2.02e-04, LoRA r: 8
   Dropout: 0.190, Steps: 52
==((====))==  Unsloth 2025.10.1: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.741 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.4.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
   🏃 Training in corso...

[52/52 01:03, Epoch 5/6]
Step 	Training Loss
5 	2.068700
10 	1.980900
15 	1.760700
20 	1.700300
25 	1.504900
30 	1.372200
35 	1.239700
40 	1.228600
45 	0.966700
50 	1.256100

   ✅ Loss: 1.4941
📊 Progresso: 12/12 | ETA: 0.0min | Tempo: 18.3min | Miglior Loss: 1.0925339525396174

============================================================
📊 RISULTATI OTTIMIZZAZIONE
============================================================
📈 Trial completati: 12/12
📉 Loss media: 1.3877 ± 0.2298
🎯 Miglior loss: 1.0925

🔍 ANALISI PARAMETRI:
   Learning Rate:
     1.1e-04: 1.3708 (n=1)
     1.2e-04: 1.2052 (n=1)
     1.3e-04: 1.6251 (n=1)
     1.5e-04: 1.1688 (n=1)
     2.0e-04: 1.4941 (n=1)
     2.5e-04: 1.5181 (n=1)
     3.3e-04: 1.2107 (n=1)
     4.6e-04: 1.0925 (n=1)
     4.9e-04: 1.1933 (n=1)
     5.2e-05: 1.4697 (n=1)
     7.1e-05: 1.9389 (n=1)
     7.2e-05: 1.3646 (n=1)
   LoRA r:
     4: 1.5732 (n=4)
     8: 1.2600 (n=3)
     16: 1.3708 (n=1)
     32: 1.3021 (n=4)

🏅 TOP 3 CONFIGURAZIONI:

#1 - Loss: 1.0925
   LR: 4.62e-04
   LoRA r: 8, α: 16
   Dropout: 0.069
   Steps: 55

#2 - Loss: 1.1688
   LR: 1.49e-04
   LoRA r: 32, α: 64
   Dropout: 0.162
   Steps: 57

#3 - Loss: 1.1933
   LR: 4.88e-04
   LoRA r: 8, α: 16
   Dropout: 0.012
   Steps: 47

🚀 CONFIGURAZIONE FINALE OTTIMIZZATA:
========================================
   lora_r: 8
   lora_alpha: 16
   lora_dropout: 0.06896194237202181
   max_steps: 300
   learning_rate: 0.0004621179133977688

💾 Configurazione salvata in 'optimized_config_simple.json'
✅ OTTIMIZZAZIONE COMPLETATA!

🎉 SUCCESSO! Configurazione ottimale trovata:
   Learning Rate: 4.62e-04
   LoRA r: 8
   Dropout: 0.069

📝 Usa questi parametri per il training completo!

  """
