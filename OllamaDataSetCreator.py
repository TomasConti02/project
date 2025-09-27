import ollama
import json
import os

def crea_ticket_strutturato(prompt, categoria_ticket="Iscrizioni Online", modello='llama3:8b', file_output='ticket.json'):
    """
    Crea un ticket strutturato con JSON costruito programmaticamente
    e aggiunge in append al file esistente
    """
    try:
        # Chiediamo solo la risposta dell'assistant al modello
        system_message = f"Sei Sara, un'assistente specializzata nella gestione di ticket. Il ticket riguarda: {categoria_ticket}. Rispondi in modo professionale e cortese."
        
        response = ollama.chat(
            model=modello,
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ]
        )
        
        risposta_assistant = response['message']['content']
        
        # Costruiamo il nuovo ticket
        nuovo_ticket = {
            "conversations": [
                {
                    "role": "system", 
                    "content": f"Sei un assistente specializzato nella gestione di ticket. Il ticket riguarda: {categoria_ticket}."
                },
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": risposta_assistant
                }
            ]
        }
        
        # Gestione del file esistente
        if os.path.exists(file_output):
            # Leggi il file esistente
            with open(file_output, 'r', encoding='utf-8') as file:
                try:
                    dati_esistenti = json.load(file)
                    # Se è una lista, aggiungi il nuovo ticket
                    if isinstance(dati_esistenti, list):
                        dati_esistenti.append(nuovo_ticket)
                    else:
                        # Se non è una lista, crea una nuova lista
                        dati_esistenti = [dati_esistenti, nuovo_ticket]
                except json.JSONDecodeError:
                    # Se il file non è JSON valido, crea una nuova lista
                    dati_esistenti = [nuovo_ticket]
        else:
            # Se il file non esiste, crea una nuova lista
            dati_esistenti = [nuovo_ticket]
        
        # Salvataggio del JSON aggiornato
        with open(file_output, 'w', encoding='utf-8') as file:
            json.dump(dati_esistenti, file, indent=2, ensure_ascii=False)
        
        print(f"✅ Ticket aggiunto in append a: {file_output}")
        print(f"📊 Totale ticket nel file: {len(dati_esistenti)}")
        return nuovo_ticket
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        return None

# Funzione per leggere tutti i ticket
def leggi_tutti_i_ticket(file_output='ticket.json'):
    """
    Legge e mostra tutti i ticket nel file
    """
    try:
        if os.path.exists(file_output):
            with open(file_output, 'r', encoding='utf-8') as file:
                tickets = json.load(file)
                print(f"📁 File: {file_output}")
                print(f"📋 Numero totale di ticket: {len(tickets)}")
                print("=" * 60)
                
                for i, ticket in enumerate(tickets, 1):
                    print(f"\n🎫 TICKET #{i}")
                    print("-" * 40)
                    user_msg = ticket['conversations'][1]['content'][:100] + "..." if len(ticket['conversations'][1]['content']) > 100 else ticket['conversations'][1]['content']
                    assistant_msg = ticket['conversations'][2]['content'][:100] + "..." if len(ticket['conversations'][2]['content']) > 100 else ticket['conversations'][2]['content']
                    print(f"👤 User: {user_msg}")
                    print(f"🤖 Assistant: {assistant_msg}")
                
                return tickets
        else:
            print("📭 File non trovato")
            return []
    except Exception as e:
        print(f"❌ Errore nella lettura: {e}")
        return []

# Test con multiple conversazioni
prompt_1 = """Buongiorno, segnalo che per il servizio di pre-scuola non sono disponibili tutte le classi. 
Mancano la 2B della scuola Rodari e la 4A della scuola Montessori. 
Potete aggiungerle? Inoltre, confermate che le iscrizioni multiple per fratelli vadano fatte in un'unica domanda? 
Grazie, Luca"""

prompt_2 = """Buonasera, ho problemi con l'accesso al portale delle iscrizioni. 
Quando inserisco le credenziali, ricevo un errore "Utente non riconosciuto". 
Potete verificare che l'account sia attivo? Il mio codice utente è UT12345. 
Grazie, Maria"""

prompt_3 = """Salve, vorrei segnalare che nel modulo di iscrizione al servizio mensa 
mancano alcune opzioni dietetiche specifiche, come l'opzione senza glutine 
e l'opzione vegetariana. È possibile aggiungerle? 
Cordiali saluti, Roberto"""

# Esegui i test
print("🚀 Creazione primo ticket...")
crea_ticket_strutturato(prompt_1, "Iscrizioni Online", 'llama3:8b', 'tickets_multi.json')

print("\n🚀 Creazione secondo ticket...")
crea_ticket_strutturato(prompt_2, "Problemi Accesso", 'llama3:8b', 'tickets_multi.json')

print("\n🚀 Creazione terzo ticket...")
crea_ticket_strutturato(prompt_3, "Moduli Iscrizione", 'llama3:8b', 'tickets_multi.json')

# Leggi e mostra tutti i ticket
print("\n" + "="*60)
leggi_tutti_i_ticket('tickets_multi.json')
