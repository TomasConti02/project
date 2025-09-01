%%capture
import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

# ----------------------------
# 1️⃣ Carica il modello
# ----------------------------
max_seq_length = 2048
dtype = torch.float16  # T4 supporta bene fp16

model_name = "unsloth/Llama-3.2-1B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=False
)

# Applica il chat template LLaMA-3.1
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# ----------------------------
# 2️⃣ LoRA setup
# ----------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False
)

# ----------------------------
# 3️⃣ Load dataset Bash
# ----------------------------
dataset = load_dataset("json", data_files="bash_dataset_100.jsonl", split="train")

# ----------------------------
# 4️⃣ Trasforma in prompt testuale
# ----------------------------
def format_instruction_prompt(example):
    # Costruisce prompt testuale stile chat LLaMA-3.1
    # User = istruzione + input, Assistant = output
    user_content = example["instruction"]
    if example["input"]:
        user_content += f"\nInput: {example['input']}"
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(format_instruction_prompt)

# ----------------------------
# 5️⃣ Prepara il trainer
# ----------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=data_collator,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none"
    )
)

# ----------------------------
# 6️⃣ Addestra solo sulle risposte (ignore user input tokens)
# ----------------------------
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# ----------------------------
# 7️⃣ Training
# ----------------------------
trainer_stats = trainer.train()

# ----------------------------
# 8️⃣ Test inferenza
# ----------------------------
user_prompt = "Crea un file vuoto chiamato esempio.txt"
messages = [{"role": "user", "content": user_prompt}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ----------------------------
# 9️⃣ Salvataggio del modello
# ----------------------------
model.save_pretrained_gguf("my_model", tokenizer, quantization_method="q4_k_m")
