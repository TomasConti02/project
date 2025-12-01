import json
import ollama
import re

def get_all_conversations(json_path):
    """
    Restituisce tutte le conversazioni dal file JSON
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            tickets = json.load(f)
    except Exception as e:
        print(f"Errore lettura JSON: {e}")
        return []

    conversations = []
    for ticket in tickets:
        compact_messages = []
        for msg in ticket["Messages"]:
            compact_body = ' '.join(msg["Body"].split())
            tipo = msg.get("Tipo", "Sconosciuto")
            poster = msg.get("Poster", "Sconosciuto")
            formatted = f"[{tipo} - {poster}]: {compact_body}"
            compact_messages.append(formatted)

        num_messages = len(ticket["Messages"])
        ticket_header = f"=== TICKET #{ticket['Number']} - {ticket['Ente']} - {ticket['Subject']} - {num_messages} messaggi ==="
        full_conversation = ticket_header + " | " + ' | '.join(compact_messages)
        conversations.append({
            'ticket_number': ticket['Number'],
            'conversation_text': full_conversation,
            'num_messages': num_messages,
            'ente': ticket['Ente'],
            'subject': ticket['Subject']
        })

    return conversations

def clean_knowledge_base_text(text):
    """
    Pulisce e formatta il testo per il RAG
    """
    # Rimuove riferimenti al sistema RAG e chatbot
    text = re.sub(r'sistema\s+RAG.*?assistenza\s+clienti', '', text, flags=re.IGNORECASE)
    text = re.sub(r'chatbot.*?assistenza\s+clienti', '', text, flags=re.IGNORECASE)
    text = re.sub(r'knowledge\s+base.*?assistenza\s+clienti', '', text, flags=re.IGNORECASE)
    text = re.sub(r'guida\s+operativa.*?assistenza', '', text, flags=re.IGNORECASE)

    # Rimuove frasi introduttive generiche
    text = re.sub(r'^(La\s+conversazione\s+riguarda|Il\s+problema\s+specifico\s+è|L\'utente\s+ha\s+richiesto).*?\.', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Rimuove linee di separazione e caratteri speciali
    text = re.sub(r'---+\s*', '', text)
    text = re.sub(r'===+\s*', '', text)
    text = re.sub(r'\*\*\*+\s*', '', text)

    # Rimuove numerazioni tipo "1." "2." ma mantiene il testo
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)

    # Rimuove spazi multipli e newline inconsistenti
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()

def save_to_txt(content, filename="knowledge_base.txt"):
    """
    Salva il contenuto in append su file txt con formattazione pulita
    """
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content + "\n" + "="*80 + "\n\n")

def generate_title_and_kb_with_ollama(conversations, output_file="knowledge_base.txt", batch_size=10):
    """
    Genera knowledge base per le ultime 10 conversazioni
    """
    results = []

    # Pulisce il file all'inizio
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("KNOWLEDGE BASE COMPLETA PER ASSISTENZA TECNICA\n")
        f.write(f"Totale conversazioni: {len(conversations)}\n")
        f.write("=" * 80 + "\n\n")

    total_conversations = len(conversations)
    processed = 0

    for i, conv in enumerate(conversations, 1):
        print(f"\n{'='*80}")
        print(f"GENERAZIONE {i}/{total_conversations}")
        print(f"Elaborazione Ticket #{conv['ticket_number']} - {conv['ente']}")
        print(f"{'='*80}")

        # PROMPT MIGLIORATO - più specifico e diretto
        prompt = f"""
Crea una guida tecnica operativa basata ESCLUSIVAMENTE su questa conversazione di supporto.

CONVERSAZIONE:
{conv['conversation_text']}

ISTRUZIONI PER LA GUIDA TECNICA:
1. DESCRIVI il problema specifico in modo dettagliato
2. ELENCA tutti i passaggi tecnici necessari per risolvere il problema
3. INCLUDI tutti i codici, parametri, valori specifici menzionati
4. SPECIFICA i nomi esatti di moduli, menu, pulsanti da utilizzare
5. USA solo "Utente" e "Operatore" come riferimenti alle persone
6. SCRIVI in stile tecnico-operativo, diretto e conciso
7. EVITA frasi introduttive generiche come "La conversazione riguarda..."

La guida deve essere immediatamente utilizzabile da un operatore per risolvere problemi identici.

ESEMPIO DI FORMATO IDEALE:
Per risolvere [problema], seguire questi passaggi: Accedere a [modulo] > Selezionare [menu] > Inserire [codice] > Cliccare [pulsante]. Utilizzare i parametri: [valore1], [valore2]. Verificare che [condizione].
"""
        try:
            print("Generazione knowledge base ottimizzata...")
            response = ollama.chat(
                model='deepseek-r1:14b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            knowledge_base_text = response['message']['content'].strip()

            # Pulisce e formatta il contenuto in modo più aggressivo
            cleaned_kb = clean_knowledge_base_text(knowledge_base_text)

            # Verifica la qualità della knowledge base
            quality_score = assess_knowledge_base_quality(cleaned_kb)

            # Crea il contenuto finale con titolo e oggetto dal dataset
            final_content = f"Titolo: {conv['subject']}\nOggetto: Assistenza per {conv['subject']} - {conv['ente']}\nQualità: {quality_score}/10\nKnowledge Base: {cleaned_kb}"

            conv['title_and_kb'] = final_content
            conv['quality_score'] = quality_score

            # Salva immediatamente nel file
            save_to_txt(final_content, output_file)

            # Mostra immediatamente il risultato
            print(f"\n✓ KNOWLEDGE BASE GENERATA - Ticket #{conv['ticket_number']}")
            print(f"📊 Qualità: {quality_score}/10")
            print(f"📝 Lunghezza: {len(cleaned_kb)} caratteri")
            print(f"✓ Salvato in: {output_file}")
            print(f"{'='*80}")

            processed += 1

            # Gestione batch - pausa ogni batch_size conversazioni
            if processed % batch_size == 0:
                print(f"\n🔄 BATCH COMPLETATO: {processed}/{total_conversations}")
                print("⏸️  Pausa di 30 secondi...")
                import time
                time.sleep(30)

        except Exception as e:
            error_msg = f"Errore generazione KB: {e}"
            conv['title_and_kb'] = error_msg
            conv['quality_score'] = 0

            # Crea contenuto di errore con titolo e oggetto dal dataset
            error_content = f"Titolo: {conv['subject']}\nOggetto: Assistenza per {conv['subject']} - {conv['ente']}\nQualità: 0/10\nKnowledge Base: Impossibile generare la knowledge base per questo ticket. Si prega di consultare il ticket originale #{conv['ticket_number']} per i dettagli completi."
            save_to_txt(error_content, output_file)

            print(f"\n✗ ERRORE Ticket #{conv['ticket_number']}")
            print(error_msg)
            print(f"{'='*80}")

        results.append(conv)

        # Progresso ogni 2 conversazioni (più frequente per solo 10 conversazioni)
        if i % 2 == 0:
            print(f"\n📈 PROGRESSO: {i}/{total_conversations} ({i/total_conversations*100:.1f}%)")

    return results

def assess_knowledge_base_quality(text):
    """
    Valuta la qualità della knowledge base generata
    """
    score = 5  # Punteggio base

    # Criteri di qualità
    if re.search(r'(accedere|selezionare|cliccare|inserire|modificare)', text, re.IGNORECASE):
        score += 2  # Contiene verbi operativi

    if re.search(r'(\d+|codice|parametro|valore|configurazione)', text, re.IGNORECASE):
        score += 2  # Contiene elementi tecnici specifici

    if len(text.split()) > 100:
        score += 1  # Lunghezza adeguata

    if not re.search(r'(conversazione|dialogo|messaggio)', text, re.IGNORECASE):
        score += 1  # Non parla della conversazione ma del contenuto

    if re.search(r'(seguire|procedere|utilizzare|verificare)', text, re.IGNORECASE):
        score += 1  # Usa linguaggio direttivo

    return min(score, 10)  # Massimo 10

def main():
    JSON_PATH = "/content/tickets_filtered.json"
    OUTPUT_FILE = "/content/drive/MyDrive/knowledge_base_ultime_10.txt"

    print("🎯 INIZIO ELABORAZIONE DELLE ULTIME 10 CONVERSAZIONI")
    print("=" * 80)

    print("Recupero tutte le conversazioni...")
    conversations = get_all_conversations(JSON_PATH)
    if not conversations:
      print("Nessuna conversazione trovata.")
      return

    # 🔹 Filtro: solo ticket con numero >= 91
    conversations = [c for c in conversations if c['ticket_number'] >= 91]

    if not conversations:
        print("Nessuna conversazione trovata.")
        return

    # 🔹 MODIFICA PRINCIPALE: prendi solo le ultime 10 conversazioni
    conversations = conversations[-10:]

    print(f"🎉 Selezionate le ULTIME {len(conversations)} conversazioni")
    print(f"📁 File di output: {OUTPUT_FILE}")
    print(f"⚙️  Batch size: {len(conversations)} conversazioni")
    print("=" * 80)

    # Mostra quali ticket verranno elaborati
    print("📋 TICKET SELEZIONATI PER L'ELABORAZIONE:")
    for i, conv in enumerate(conversations, 1):
        print(f"   {i}. Ticket #{conv['ticket_number']} - {conv['ente']} - {conv['subject']}")
    print("=" * 80)

    # Stima tempo di elaborazione (approssimativa)
    estimated_time = len(conversations) * 30 / 60  # 30 secondi per conversazione
    print(f"⏱️  Tempo stimato: {estimated_time:.1f} minuti")

    final_results = generate_title_and_kb_with_ollama(conversations, OUTPUT_FILE, batch_size=len(conversations))

    # Riassunto finale con statistiche dettagliate
    print(f"\n{'='*80}")
    print("✅ ELABORAZIONE COMPLETATA!")
    print(f"📁 File generato: {OUTPUT_FILE}")
    print(f"📊 Knowledge base create: {len(final_results)}")

    # Calcola statistiche di qualità
    successful = [r for r in final_results if "Errore" not in r['title_and_kb']]
    quality_scores = [r.get('quality_score', 0) for r in successful]

    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        high_quality = sum(1 for score in quality_scores if score >= 7)
        excellent_quality = sum(1 for score in quality_scores if score >= 9)

        print(f"\n📈 STATISTICHE DI QUALITÀ DETTAGLIATE:")
        print(f"   • Qualità media: {avg_quality:.1f}/10")
        print(f"   • Knowledge base totali: {len(final_results)}")
        print(f"   • Knowledge base con successo: {len(successful)}")
        print(f"   • Knowledge base di alta qualità (≥7): {high_quality}")
        print(f"   • Knowledge base eccellenti (≥9): {excellent_quality}")
        print(f"   • Knowledge base con errori: {len(final_results) - len(successful)}")
        print(f"   • Tasso di successo: {len(successful)/len(final_results)*100:.1f}%")

    print(f"{'='*80}")

    # Anteprima del file
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"\n📋 ANTEPRIMA DEL FILE FINALE:")
            print("=" * 60)
            lines = content.split('\n')
            # Mostra tutto il contenuto (solo 10 knowledge base)
            for line in lines:
                if line.strip():
                    print(line)
            print("=" * 60)

            # Statistiche file
            file_size = len(content) / 1024 / 1024  # MB
            kb_count = content.count('Titolo:')
            print(f"\n💾 DIMENSIONI FILE: {file_size:.2f} MB")
            print(f"📄 KNOWLEDGE BASE NEL FILE: {kb_count}")

    except Exception as e:
        print(f"Errore nella lettura del file: {e}")

if __name__ == "__main__":
    main()
