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
dtype = None              # Automatically selects float16 or bfloat16 depending on GPU support
#load_in_4bit = True       # Load model in 4-bit quantization (saves VRAM)
# List of 4-bit quantized models supported by Unsloth
dtype = torch.float16        # Precisione alta, ideale su T4
#load_in_4bit = False         # Disabilitiamo la quantizzazione
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
    #model_name = "unsloth/Llama-3.2-3B-Instruct",  #il modello
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length, #max seq d token (max context lenght)
    dtype = dtype, #data type 16bit more accuracy
    load_in_4bit = False, 
    # token = "hf_..."  # Uncomment if your model requires authentication
)
# model → è l’oggetto PyTorch che rappresenta la rete neurale (LLM).
# tokenizer → converte il testo in token numerici e viceversa, fondamentale per dialogare col modello.
##################################################################################################
# Apply Parameter-Efficient Fine-Tuning (PEFT) using LoRA
#Prende il modello base e lo “avvolge” con gli adapter LoRA (le due matrici low-rank per ogni layer target). Il modello originale resta congelato; alleni solo gli adapter.
#LoRA sarebbe un modo per fare il fine tuning aggiungendo parametri al modello, NON RIFACCIO L'INTERA neural-network
#ritrasformiamo il modello in uno adatto per LoRA COME ? aggiungiamo layer di matrici per tuning
"""
r = 16 -> LoRA rank: la “capacità” dell’adapter.
Più è alto → più parametri negli adapter → più capacità di adattamento, ma anche più VRAM/tempo.
Regola pratica:
task semplice / dataset piccolo: 4–8
task medio: 8–16
task difficile / stile molto diverso: 16–32
Effetto collaterale: r↑ aumenta il rischio di overfitting se i dati sono pochi → usa lora_dropout > 0.
MAGARI PROVIAMO A RIDURLO
-----------------------------------------------------------------------------------------------------------
lora_dropout = 0 -> Dropout applicato solo al ramo LoRA durante il training (regularization).
Con 0 → massima capacità, ma più rischio di overfitting su dataset piccoli.
Consigli:
dataset piccolo/rumoroso: 0.05–0.1
dataset medio/grande: 0–0.05
----------------------------------------------------------------------------------------------------------
use_rslora = False -> Rank-Stabilized LoRA: variante che tenta di stabilizzare il rango effettivo e la scala durante l’allenamento (mitiga “rank collapse”).
False va benissimo in molti casi. Se noti instabilità o drift, puoi provare True (a costo di un po’ di complessità in più).

"""
model = FastLanguageModel.get_peft_model(
    model,
    #r = LoRA rank (quanto “potere” di adattamento ha; di solito 8, 16, 32), NON mettiamo Bill di parametri in più ma Million, meno havy
    r = 16,  # LoRA rank (higher = more capacity but slower) sarebbe la capacità dell'adapter che vado ad aggiungere 
    target_modules = [ #Quali sotto-matrici del Trasformer adattare.
        "q_proj", "k_proj", "v_proj", "o_proj", #proiezioni dell’attenzione (Q/K/V + output).
        "gate_proj", "up_proj", "down_proj" #MLP del blocco Trasformer.
    ],
    lora_alpha =  16,        # Scaling factor for LoRA
    lora_dropout = 0,          # No dropout for training stability
    bias = "none",             # No additional bias parameters
    use_gradient_checkpointing = "unsloth",  # Reduces VRAM usage during training
    random_state = 3407,       # Seed for reproducibility
    use_rslora = False,        # Disable Rank-Stabilized LoRA
    loftq_config = None        # NO QUANT ADAPTER, Not using LoftQ quantization LoftQ = tecnica per combinare quantizzazione e LoRA in modo consapevole (utile se vuoi 4-bit + LoRA con meno perdita).
)
##################################################################################################
"""
Ogni modello LLM ha un formato diverso per le chat:
come scrivere i messaggi dell’utente,
come delimitare la risposta dell’assistente,
quali special tokens usare (<|user|>, <|assistant|>, <s>, ecc).
Se non usi il formato giusto, il modello non capisce più se sta leggendo un prompt dell’utente o se deve generare una risposta.
Prende il tuo tokenizer originale e gli dice:
“Quando preparo le conversazioni, usa la sintassi stile LLaMA-3.1”.
In pratica aggiunge al tokenizer la funzione apply_chat_template, che sa come convertire una lista di messaggi in un prompt testuale (o in token) compatibile con il modello.
"""
from unsloth.chat_templates import get_chat_template
#assegno al token il llama-3.1 template
tokenizer = get_chat_template( #in pratica dico al tokenizer il formato del prompt del chatbot e lo restituisco
    tokenizer,
    chat_template = "llama-3.1",
)
##################################################################################################
#ora cerchiamo di adattare il dataset grezzo al tuning
#dataset organizzato in conversations 
"""
per ogni conversations (convo) -> formata da tanti domanda e risposta user e chatbot
apply_chat_template → converte la lista di messaggi in testo formattato nello stile che il modello si aspetta (nel tuo caso llama-3.1).
tokenize=False → ti restituisce una stringa di testo, non direttamente token.
add_generation_prompt=False → NON aggiunge il tag speciale che segnala al modello di generare la risposta (utile durante addestramento; durante inference invece lo metti a True).
return {"text": texts} → crea una nuova colonna "text" che contiene le conversazioni già
Trasformo 
{
  "conversations": [
    {"role": "user", "content": "ciao"},
    {"role": "assistant", "content": "ciao! come stai?"}
  ]
}
In 
<|start_header_id|>user<|end_header_id|>
ciao
<|start_header_id|>assistant<|end_header_id|>
ciao! come stai?
"""
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass
#faccio il load del dataset personalizzato 
from datasets import load_dataset
#dataset = load_dataset("tomasconti/TestTuning", split = "train")
dataset = load_dataset(
    path = "tomasconti/TestTuning",                 # HF dataset repo or local directory
    data_files = "Test2.json",  # JSON file with formatted conversations
    split = "train"                                 # Load the training split
)
##################################################################################################
from datasets import Dataset #facciamo delle print check per vedere come è venuto fuori il load 
dataset.to_pandas().head()
print(dataset)
print(dataset[0])  # Primo elemento per vedere la struttura
print(dataset[1])
print(dataset[2])
##################################################################################################
"""
Serve a uniformare il dataset a un formato compatibile con le chat template (conversations).
I dataset tipo ShareGPT o simili spesso hanno strutture diverse -> standardize_sharegpt prende tutte queste varianti e le trasforma in uno schema unico

.map() è un metodo dei dataset Hugging Face → applica una funzione a tutti gli esempi.
Qui applichi la tua formatting_prompts_func, cioè la funzione che prende "conversations" e lo trasforma in testo già pronto col chat template LLaMA-3.1.
batched=True significa che la funzione riceve un batch di esempi alla volta (un dizionario di liste), non un solo record.

l dataset ora ha una colonna "text" con le conversazioni formattate esattamente come il modello si aspetta.
"""
from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)
##################################################################################################
"""
SFTTrainer: viene da trl (Transformers Reinforcement Learning), è una classe comoda che estende Trainer di Hugging Face per fare fine-tuning supervisionato su dataset in stile chat/istruzioni.
TrainingArguments: contiene tutti gli iperparametri e le configurazioni per l’addestramento (batch size, learning rate, logging, ecc).
DataCollatorForSeq2Seq: gestisce la preparazione dei batch (padding, labels per la loss, ecc).
is_bfloat16_supported(): funzione di unsloth che controlla se la tua GPU supporta bfloat16 (importante per usare meno memoria e avere training più stabile rispetto a fp16).
"""
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
"""
preparo il training loop -> SFTTrainer (Supervised Fine-Tuning Trainer) per addestrare il tuo modello con i dati preparati
model → il modello di base che stai fine-tunando (es. LLaMA 3.1, Mistral, ecc).
tokenizer → il tokenizer associato al modello (devono essere compatibili).
train_dataset → il dataset formattato (quello che abbiamo passato per standardize_sharegpt + .map()), deve contenere la colonna "text".
dataset_text_field="text" → dice al trainer quale colonna del dataset usare.
max_seq_length → lunghezza massima della sequenza (i testi più lunghi vengono troncati).
data_collator → funzione che costruisce i batch (applica padding, genera i label, ecc).
dataset_num_proc=2 → numero di processi paralleli per il preprocessing (velocizza).
packing=False → se fosse True, concatena esempi corti in sequenze più lunghe (più efficiente, ma può rovinare i dati tipo chat, quindi di solito lo lasci False).
"""
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01, #regolarizzazione per evitare overfitting.
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)
"""
Prendi il mio modello, questo tokenizer, il dataset in text, addestralo per 60 step con 
batch effettivo 8, ottimizzatore AdamW 8-bit, learning rate 2e-4, salva in outputs, e usa precisione fp16/bf16 a seconda della GPU
"""
##################################################################################################
"""
Quella riga sta dicendo al tuo trainer di calcolare la loss solo sulle risposte del 
modello (assistant) e non sulle parti dell’utente.
Durante il fine-tuning, di default il modello impara a predire ogni token, 
anche quelli del prompt dell’utente.
Con train_on_responses_only, i token che fanno parte dell’utente vengono 
ignorati nella loss (mascherati).
In pratica, il modello impara solo a generare la risposta, dato il prompt già corretto.
Questo:
Riduce il rumore nel training (non serve insegnare al modello a riscrivere la parte 
dell’utente).
Aumenta la qualità delle risposte e accelera il training.
"""
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
##################################################################################################
space = tokenizer(" ", add_special_tokens=False).input_ids[0]
sample_idx = 0  # oppure 1 o 2
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[sample_idx]["labels"]])
##################################################################################################
trainer_stats = trainer.train()
##################################################################################################
messages = [
    {"role": "user", "content": "descrivimi il voucher-code aziendale?"}
]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
##################################################################################################
# Salvataggio

model.save_pretrained_gguf("my_model", tokenizer, quantization_method="q4_k_m")
