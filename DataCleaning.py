import pandas as pd
import re
import json
from typing import List, Dict

def clean_dataset_enhanced(input_file, output_file=None):
    """Pulizia dataset con pulizie approfondite"""
    df = pd.read_csv(input_file, sep=';', encoding='utf-8', low_memory=False)
    print(f"Originale: {df.shape[0]} righe, {df.shape[1]} colonne")
    # Colonne utili
    useful_columns = ['topic', 'number', 'id', 'type', 'poster', 'title', 'body', 'created']
    useful_columns = [col for col in useful_columns if col in df.columns]
    df = df[useful_columns].copy()
    # Pulizie di base
    df['topic'] = df['topic'].str.strip().fillna('Altro')
    df['poster'] = df['poster'].str.strip().fillna('Utente sconosciuto')
    df['title'] = df['title'].replace(['NULL', 'NaN', '', 'x assistenza'], 'Nessun titolo').fillna('Nessun titolo')
    df['created'] = pd.to_datetime(df['created'], format='%d/%m/%Y %H:%M', errors='coerce')
    # Pulizia testo
    def clean_text(text):
        if pd.isna(text): return ''
        text = re.sub(r'<[^>]+>', ' ', str(text))
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text.strip()
    df['body_clean'] = df['body'].apply(clean_text)
    # Rimuovi righe vuote
    initial_count = len(df)
    df = df[df['body_clean'].str.len() > 10].copy()
    removed_count = initial_count - len(df)
    print(f"Pulito: {df.shape[0]} righe (-{removed_count} vuote)")
    # Metriche
    df['body_length'] = df['body_clean'].str.len()
    df['word_count'] = df['body_clean'].str.split().str.len()
    if output_file:
        df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
        print(f"Salvato in {output_file}")
    return df

def sort_dataset_by_ticket_number(df, output_file=None):
    """Ordina per numero ticket e ora di creazione, rimuove colonne non necessarie"""
    # Converti la colonna 'created' in datetime
    if 'created' in df.columns:
        df['created'] = pd.to_datetime(df['created'], errors='coerce')
        # Ordina per numero ticket e poi per data di creazione
        df_sorted = df.sort_values(by=['number', 'created']).reset_index(drop=True)
        print("Ordinamento effettuato per: numero ticket e data di creazione (created)")
    else:
        # Fallback: ordina solo per numero ticket
        df_sorted = df.sort_values(by='number').reset_index(drop=True)
        print("Ordinamento effettuato solo per numero ticket (colonna 'created' non trovata)")
    
    # Colonne da eliminare
    columns_to_drop = ['id', 'poster', 'title', 'created', 'body_length']
    existing_columns = [col for col in columns_to_drop if col in df_sorted.columns]
    if existing_columns:
        df_sorted = df_sorted.drop(columns=existing_columns)
        print(f"Colonne eliminate: {existing_columns}")
    
    if output_file:
        # Salva con righe di separazione tra ticket diversi
        with open(output_file, 'w', encoding='utf-8') as f:
            # Scrivi l'header
            f.write(';'.join(df_sorted.columns) + '\n')
            # Scrivi i dati con separatori
            prev_ticket = None
            for i, row in df_sorted.iterrows():
                current_ticket = row['number']
                # Se è un ticket nuovo, aggiungi riga separatrice
                if prev_ticket is not None and current_ticket != prev_ticket:
                    f.write(';;;' + '\n')  # Riga separatrice
                # Scrivi la riga normale
                row_values = [str(x) if pd.notna(x) else '' for x in row]
                f.write(';'.join(row_values) + '\n')
                prev_ticket = current_ticket
        print(f"Dataset ordinato salvato in: {output_file}")
    
    return df_sorted

def create_conversations_json_from_df(df: pd.DataFrame, output_file: str = "conversations.json") -> List[Dict]:
    """
    Crea un file JSON con le conversazioni dal DataFrame (senza filtri).
    """
    conversations = []
    current_ticket = None
    current_messages = []
    ticket_count = 0
    
    for i, row in df.iterrows():
        # Salta le righe separatrice (se presenti nel DataFrame)
        if pd.isna(row['number']) or row['number'] == '':
            continue
        
        ticket_number = row['number']
        
        # Se è un nuovo ticket e abbiamo messaggi precedenti, crea la conversazione
        if current_ticket is not None and ticket_number != current_ticket and current_messages:
            conversation_entry = {
                "conversations": current_messages
            }
            conversations.append(conversation_entry)
            current_messages = []
            ticket_count += 1
        
        current_ticket = ticket_number
        # Determina il ruolo in base al tipo (M = user, R = assistant)
        role = "user" if row['type'] == 'M' else "assistant"
        # Crea il messaggio
        message = {
            "role": role,
            "content": row['body_clean']
        }
        
        # Se è il primo messaggio del ticket, aggiungi il system message
        if not current_messages:
            system_message = {
                "role": "system", 
                "content": f"Sei un assistente specializzato nella gestione di ticket. Il ticket riguarda: {row['topic']}."
            }
            current_messages.append(system_message)
        
        current_messages.append(message)
    
    # Aggiungi l'ultima conversazione
    if current_messages:
        conversation_entry = {
            "conversations": current_messages
        }
        conversations.append(conversation_entry)
        ticket_count += 1
    
    # Salva il JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Creato file {output_file} con {len(conversations)} conversazioni (senza filtri)")
    return conversations

def is_valid_conversation(messages: List[Dict]) -> bool:
    """
    Valida se una conversazione è adatta per il training.
    """
    if len(messages) < 3:  # system + user + assistant minimo
        return False
    
    # Controlla che inizi con user dopo system
    if len(messages) >= 2 and messages[1]['role'] != 'user':
        return False
    
    # Controlla alternanza user/assistant
    user_count = sum(1 for msg in messages if msg['role'] == 'user')
    assistant_count = sum(1 for msg in messages if msg['role'] == 'assistant')
    
    if user_count == 0 or assistant_count == 0:
        return False
    
    # Controlla lunghezza minima dei contenuti
    for msg in messages:
        if len(msg['content'].strip()) < 5:
            return False
    
    return True

def create_conversations_json_improved_from_df(df: pd.DataFrame, output_file: str = "conversations_clean.json") -> List[Dict]:
    """
    Crea un file JSON con conversazioni validate e filtrate dal DataFrame.
    """
    conversations = []
    current_ticket = None
    current_messages = []
    ticket_count = 0
    skipped_tickets = 0
    
    for i, row in df.iterrows():
        if pd.isna(row['number']) or row['number'] == '':
            continue
            
        ticket_number = row['number']
        
        # Se è un nuovo ticket, valida e salva il precedente
        if current_ticket is not None and ticket_number != current_ticket:
            if is_valid_conversation(current_messages):
                conversation_entry = {
                    "conversations": current_messages
                }
                conversations.append(conversation_entry)
                ticket_count += 1
            else:
                skipped_tickets += 1
                
            current_messages = []
        
        current_ticket = ticket_number
        role = "user" if row['type'] == 'M' else "assistant"
        
        message = {
            "role": role,
            "content": row['body_clean']
        }
        
        # System message solo per primo messaggio valido
        if not current_messages and role == "user":
            system_message = {
                "role": "system", 
                "content": f"Sei un assistente specializzato nella gestione di ticket. Il ticket riguarda: {row['topic']}."
            }
            current_messages.append(system_message)
        
        current_messages.append(message)
    
    # Ultimo ticket
    if current_messages and is_valid_conversation(current_messages):
        conversation_entry = {
            "conversations": current_messages
        }
        conversations.append(conversation_entry)
        ticket_count += 1
    else:
        skipped_tickets += 1
    
    # Salva il JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Creato file {output_file} con {len(conversations)} conversazioni valide")
    print(f"❌ Conversazioni scartate: {skipped_tickets}")
    print(f"📈 Tasso di successo: {(len(conversations)/(len(conversations)+skipped_tickets))*100:.1f}%")
    
    return conversations

def process_ticket_dataset_optimized(input_file: str, sorted_output_file: str = None):
    """
    Processa il dataset una sola volta e genera entrambi i JSON.
    """
    # Pulizia dataset
    print("=== PULIZIA DATASET ===")
    df_clean = clean_dataset_enhanced(input_file)
    print(f"Ticket unici: {df_clean['number'].nunique()}, Messaggi totali: {len(df_clean)}")
    
    # Ordina il dataset UNA SOLA VOLTA
    print("\n=== ORDINAMENTO DATASET ===")
    df_sorted = sort_dataset_by_ticket_number(df_clean, sorted_output_file)
    
    # Crea entrambi i JSON dallo stesso DataFrame ordinato
    print("\n=== CREAZIONE JSON BASE ===")
    conversations_base = create_conversations_json_from_df(
        df_sorted, 
        "conversations_base.json"
    )
    print("\n=== CREAZIONE JSON FILTRATO ===")
    conversations_filtered = create_conversations_json_improved_from_df(
        df_sorted, 
        "conversations_filtered.json"
    )
    return df_sorted, conversations_base, conversations_filtered

if __name__ == "__main__":
    input_file = "ticket_2024_topic.csv"
    sorted_output_file = "ticket_2024_sorted.csv"
    
    # Processo ottimizzato: un solo ordinamento
    df_sorted, conversations_base, conversations_filtered = process_ticket_dataset_optimized(
        input_file, sorted_output_file
    )
    
    # Confronto finale
    print(f"\n🎯 PROCESSO COMPLETATO!")
    print(f"   Dataset finale: {len(df_sorted)} righe")
    print(f"   JSON base: {len(conversations_base)} conversazioni")
    print(f"   JSON filtrato: {len(conversations_filtered)} conversazioni")
    print(f"   Conversazioni scartate: {len(conversations_base) - len(conversations_filtered)}")
    print(f"   Qualità dataset: {(len(conversations_filtered)/len(conversations_base))*100:.1f}%")
