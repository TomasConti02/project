import ollama
import json
import re
import time

def dividi_testo_in_chunks_ottimali(testo, max_caratteri=600, min_caratteri=200):
    """
    Versione ottimizzata per il training
    """
    paragrafi = [p.strip() for p in testo.split('\n\n') if p.strip() and len(p.strip()) > 20]
    
    chunks = []
    chunk_corrente = ""
    
    for paragrafo in paragrafi:
        if len(paragrafo) > max_caratteri:
            if chunk_corrente:
                chunks.append(chunk_corrente)
                chunk_corrente = ""
            
            frasi = re.split(r'[.!?]+', paragrafo)
            frasi = [f.strip() for f in frasi if f.strip() and len(f.strip()) > 10]
            
            for frase in frasi:
                if len(chunk_corrente) + len(frase) + 2 <= max_caratteri:
                    chunk_corrente = chunk_corrente + ". " + frase if chunk_corrente else frase
                else:
                    if chunk_corrente:
                        chunks.append(chunk_corrente)
                    chunk_corrente = frase
        else:
            lunghezza_potenziale = len(chunk_corrente) + len(paragrafo) + 2 if chunk_corrente else len(paragrafo)
            
            if lunghezza_potenziale > max_caratteri and chunk_corrente:
                chunks.append(chunk_corrente)
                chunk_corrente = paragrafo
            else:
                chunk_corrente = chunk_corrente + "\n\n" + paragrafo if chunk_corrente else paragrafo
    
    if chunk_corrente:
        chunks.append(chunk_corrente)
    
    # Unisci chunks troppo piccoli
    chunks_finali = []
    for chunk in chunks:
        if len(chunk) < min_caratteri and chunks_finali:
            if len(chunks_finali[-1]) + len(chunk) + 2 <= max_caratteri:
                chunks_finali[-1] += "\n\n" + chunk
            else:
                chunks_finali.append(chunk)
        else:
            chunks_finali.append(chunk)
    
    return chunks_finali

def genera_domanda_risposta(chunk_testo, modello='llama3.1:8b', temperature=0.7, top_p=0.9, top_k=40, max_tokens=500, contesto_extra=""):
    """
    Genera una domanda e risposta basata sul chunk di testo con parametri configurabili
    """
    try:
        system_message = """Sei un assistente specializzato nella creazione di domande e risposte 
        basate su manuali tecnici. Crea UNA sola domanda pertinente che un utente potrebbe fare 
        dopo aver letto il testo, e fornisci UNA risposta accurata e completa."""
        
        # Aggiungi contesto extra se fornito
        contesto_intro = ""
        if contesto_extra:
            contesto_intro = f"\nCONTESTO: {contesto_extra}\n"
        
        prompt = f"""Crea UNA domanda specifica e la relativa risposta basata sul seguente testo:{contesto_intro}

        TESTO:
        {chunk_testo}

        Formato richiesto:
        DOMANDA: [domanda specifica sul contenuto]
        RISPOSTA: [risposta dettagliata e accurata]"""

        response = ollama.chat(
            model=modello,
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'num_predict': max_tokens,
                'seed': 42,  # Per riproducibilità
                'repeat_penalty': 1.1  # Penalità per ripetizioni
            }
        )
        
        return response['message']['content']
        
    except Exception as e:
        print(f"Errore nella generazione Q&A: {e}")
        return None

def crea_dataset_manualistica(file_input, file_output='dataset_ecivis.json', contesto_sistema="", **generazione_kwargs):
    """
    Crea il dataset usando i chunks ottimali con parametri configurabili
    """
    try:
        with open(file_input, 'r', encoding='utf-8') as file:
            testo_completo = file.read()
        
        chunks = dividi_testo_in_chunks_ottimali(testo_completo)
        print(f"Creati {len(chunks)} chunks ottimali")
        
        dataset = []
        
        for i, chunk in enumerate(chunks):
            print(f"Generando Q&A per chunk {i+1}/{len(chunks)}...")
            
            # USA SEMPRE LO STESSO CONTESTO PER TUTTI I CHUNK
            contesto_aggiuntivo = contesto_sistema  # Sempre il contesto principale
            
            qa_generato = genera_domanda_risposta(chunk, contesto_extra=contesto_aggiuntivo, **generazione_kwargs)
            
            if qa_generato:
                # Estrazione semplificata
                lines = qa_generato.split('\n')
                domanda = ""
                risposta = ""
                
                for line in lines:
                    if line.lower().startswith('domanda:') or line.lower().startswith('q:'):
                        domanda = line.split(':', 1)[1].strip()
                    elif line.lower().startswith('risposta:') or line.lower().startswith('a:'):
                        risposta = line.split(':', 1)[1].strip()
                
                if not domanda or not risposta:
                    domanda = "Domanda sul contenuto del manuale"
                    risposta = qa_generato
                
                # Crea la conversazione con il messaggio di sistema
                conversazione = {
                    "conversations": [
                        {
                            "role": "system", 
                            "content": f"Sei un assistente specializzato nella gestione di {contesto_aggiuntivo}."
                        },
                        {"role": "user", "content": domanda},
                        {"role": "assistant", "content": risposta}
                    ]
                }
                
                dataset.append(conversazione)
                print(f"✓ Chunk {i+1} completato - Contesto: {contesto_aggiuntivo}")
            
            time.sleep(1)  # Pausa per non sovraccaricare
        
        with open(file_output, 'w', encoding='utf-8') as file:
            json.dump(dataset, file, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset creato con {len(dataset)} conversazioni!")
        return dataset
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        return None

# Esempi di utilizzo con diversi setting
if __name__ == "__main__":
    # CONTESTO UNICO per tutte e tre le configurazioni
    CONTESTO_UNICO = "Assistenza Clienti al portale eCivisWeb"
    
    # Configurazione 1: Generazione conservativa (più prevedibile)
    print("=== Configurazione Conservativa ===")
    crea_dataset_manualistica(
        'test.txt',
        'dataset_conservativo.json',
        contesto_sistema=CONTESTO_UNICO,  # Stesso contesto per tutti
        temperature=0.3,      # Bassa temperatura = meno creatività
        top_p=0.8,           # Più focalizzato
        top_k=20,            # Meno scelte possibili
        max_tokens=300       # Risposte più concise
    )
    
    time.sleep(2)
    
    # Configurazione 2: Generazione bilanciata
    print("\n=== Configurazione Bilanciata ===")
    crea_dataset_manualistica(
        'test.txt',
        'dataset_bilanciato.json',
        contesto_sistema=CONTESTO_UNICO,  # Stesso contesto per tutti
        temperature=0.7,      # Medio-alta creatività
        top_p=0.9,           # Buon bilanciamento
        top_k=40,            # Ampia scelta
        max_tokens=500       # Lunghezza media
    )
    
    time.sleep(2)
    
    # Configurazione 3: Generazione creativa
    print("\n=== Configurazione Creativa ===")
    crea_dataset_manualistica(
        'test.txt',
        'dataset_creativo.json',
        contesto_sistema=CONTESTO_UNICO,  # Stesso contesto per tutti
        temperature=1.0,      # Alta creatività
        top_p=0.95,          # Ampio spettro di scelte
        top_k=60,            # Molte opzioni possibili
        max_tokens=800       # Risposte più dettagliate
    )
