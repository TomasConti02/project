import pandas as pd
import re

def clean_dataset_enhanced(input_file, output_file=None):
    """
    Versione migliorata con pulizie più approfondite
    """
    print("🧹 Pulizia dataset in corso...")
    
    # Carica il CSV
    df = pd.read_csv(input_file, sep=';', encoding='utf-8', low_memory=False)
    print(f"Dataset originale: {df.shape[0]} righe, {df.shape[1]} colonne")
    
    # Colonne minime utili
    useful_columns = ['topic', 'number', 'id', 'type', 'poster', 'title', 'body', 'created']
    useful_columns = [col for col in useful_columns if col in df.columns]
    
    df = df[useful_columns].copy()
    
    # Pulizie di base
    df['topic'] = df['topic'].str.strip().fillna('Altro')
    df['poster'] = df['poster'].str.strip().fillna('Utente sconosciuto')
    df['title'] = df['title'].replace(['NULL', 'NaN', '', 'x assistenza'], 'Nessun titolo').fillna('Nessun titolo')
    df['created'] = pd.to_datetime(df['created'], format='%d/%m/%Y %H:%M', errors='coerce')
    
    # Pulizia corpo del messaggio
    def clean_text(text):
        if pd.isna(text):
            return ''
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
    
    print(f"Dataset pulito: {df.shape[0]} righe (rimosse {removed_count} vuote)")
    
    # Metriche aggiuntive
    df['body_length'] = df['body_clean'].str.len()
    df['word_count'] = df['body_clean'].str.split().str.len()
    
    # Salva se richiesto
    if output_file:
        df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
        print(f"Salvato in {output_file}")
    
    return df

def sort_dataset_by_ticket_number(input_file, output_file=None):
    """
    Ordina il dataset per numero del ticket
    """
    df = pd.read_csv(input_file, sep=';', encoding='utf-8', low_memory=False)
    df['created'] = pd.to_datetime(df['created'], errors='coerce')
    df_sorted = df.sort_values(by='number').reset_index(drop=True)
    
    if output_file:
        df_sorted.to_csv(output_file, sep=';', index=False, encoding='utf-8')
        print(f"Dataset ordinato salvato in {output_file}")
    
    return df_sorted

def safe_date_format(date_value):
    """Formatta la data in modo sicuro"""
    if pd.isna(date_value):
        return 'N/A'
    try:
        return date_value.strftime('%d/%m/%Y %H:%M')
    except (AttributeError, ValueError):
        return 'Data non valida'

def print_tickets_with_users(df, num_rows=20):
    """
    Stampa ticket con utenti in formato compatto
    """
    print(f"\n🎯 PRIMI {num_rows} MESSAGGI (ticket + utente):")
    print("="*80)
    
    for i, (idx, row) in enumerate(df.head(num_rows).iterrows(), 1):
        ticket = row['number']
        user = row['poster'][:20]  # Tronca nomi lunghi
        msg_type = row['type']
        date = safe_date_format(row['created'])
        preview = row['body_clean'][:35] + "..." if len(row['body_clean']) > 35 else row['body_clean']
        
        print(f"{ticket:<8} {user:<20} {msg_type} {date:<16} {preview}")

def print_extended_ticket_examples(df, num_examples=15):
    """
    Stampa esempi estesi dei ticket con tutti i dettagli
    """
    print(f"\n🔍 ESEMPI TICKET DETTAGLIATI ({num_examples} conversazioni):")
    print("="*70)
    
    # Prendi i primi N ticket unici
    first_tickets = df['number'].drop_duplicates().head(num_examples)
    
    for i, ticket_num in enumerate(first_tickets, 1):
        # Filtra tutti i messaggi di questo ticket
        ticket_messages = df[df['number'] == ticket_num]
        num_messages = len(ticket_messages)
        users = ticket_messages['poster'].unique()
        topic = ticket_messages['topic'].iloc[0]
        
        # Formatta lista utenti
        if len(users) <= 3:
            user_list = ', '.join(users)
        else:
            user_list = ', '.join(users[:3]) + f"... (+{len(users)-3})"
        
        # Data inizio e fine
        start_date = safe_date_format(ticket_messages['created'].min())
        end_date = safe_date_format(ticket_messages['created'].max())
        
        print(f"{i:2d}. #{ticket_num}:")
        print(f"    • Messaggi: {num_messages}")
        print(f"    • Utenti: {user_list}")
        print(f"    • Topic: {topic}")
        print(f"    • Periodo: {start_date} - {end_date}")

def analyze_first_200_tickets(df):
    """
    Analisi compatta dei primi 200 ticket
    """
    first_200_tickets = df['number'].drop_duplicates().head(200)
    df_first_200 = df[df['number'].isin(first_200_tickets)]
    
    print(f"\n📊 ANALISI PRIMI 200 TICKET:")
    print(f"• Ticket unici: {len(first_200_tickets)}")
    print(f"• Messaggi totali: {len(df_first_200)}")
    print(f"• Media messaggi/ticket: {len(df_first_200)/len(first_200_tickets):.1f}")
    print(f"• Utenti coinvolti: {df_first_200['poster'].nunique()}")

# Uso principale
if __name__ == "__main__":
    input_file = "ticket_2024_topic.csv"
    output_file = "ticket_2024_clean_enhanced.csv"
    
    # Pulizia dataset
    df_clean = clean_dataset_enhanced(input_file, output_file)
    
    # Statistiche base
    print(f"\n📈 DATASET COMPLETO:")
    print(f"Ticket totali: {df_clean['number'].nunique()}")
    print(f"Messaggi totali: {len(df_clean)}")
    print(f"Periodo: {df_clean['created'].min().strftime('%d/%m/%Y')} - {df_clean['created'].max().strftime('%d/%m/%Y')}")
    
    # Ordina dataset
    df_sorted = sort_dataset_by_ticket_number(output_file)
    
    # Stampa anteprima compatta dei messaggi
    print_tickets_with_users(df_sorted, num_rows=20)
    
    # Stampa esempi ticket estesi
    print_extended_ticket_examples(df_sorted, num_examples=15)
    
    # Analisi primi 200 ticket
    analyze_first_200_tickets(df_sorted)
    
    # Salva lista ticket
    first_200 = df_sorted['number'].drop_duplicates().head(200).tolist()
    print(f"\n💾 Primi 200 ticket salvati in lista (totale: {len(first_200)})")
