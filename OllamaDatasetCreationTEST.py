"""
Generatore Dataset di Qualità per Fine-Tuning eCivisWeb
Versione corretta con focus su accuratezza e riduzione allucinazioni
"""

import ollama
import json
import re
import time
from typing import List, Dict, Tuple

# =============================================================================
# CONFIGURAZIONE GLOBALE
# =============================================================================

CONTESTO_SISTEMA = "Assistenza Clienti al portale eCivisWeb"

# Template di domande reali che gli utenti farebbero
DOMANDE_TEMPLATE = [
    # Accesso e autenticazione
    "Come posso accedere al portale?",
    "Quali credenziali servono per il login?",
    "Dove cambio la mia password?",
    "Non riesco ad entrare nel mio account",
    "Come faccio ad autenticarmi?",
    
    # Navigazione e struttura
    "Quali sezioni vedo senza login?",
    "Dove trovo i documenti?",
    "Dove sono i manuali?",
    "Come funziona il menu del portale?",
    "Quali moduli sono disponibili?",
    
    # Utenti e figli
    "Dove vedo i dati di mio figlio?",
    "Come controllo a quali servizi è iscritto?",
    "Cosa significa il codice tessera?",
    "Come verifico i dati anagrafici?",
    
    # Refezione scolastica
    "Come ricarico il conto mensa?",
    "Dove vedo il saldo della mensa?",
    "Come consulto i pasti consumati?",
    "Cosa sono le fasce sociali?",
    
    # Prenotazioni e disdette
    "Come disdico il pasto?",
    "Posso prenotare il pasto in bianco?",
    "Cosa significano i colori nel calendario?",
    "Devo confermare la presenza?",
    
    # Pagamenti
    "Come pago con PagoPA?",
    "Differenza tra prepagato e postpagato?",
    "Come pago le rette?",
    "Dove scarico le ricevute?",
    
    # Moduli online
    "Come presento una domanda?",
    "Posso modificare una domanda inviata?",
    "Dove vedo lo stato della mia domanda?",
    "Come cancello una domanda?",
]

# =============================================================================
# FUNZIONI MIGLIORATE
# =============================================================================

def estrai_sezioni_documento(testo: str) -> Dict[str, str]:
    """
    Divide il documento in sezioni logiche basate sui titoli
    Evita chunks troppo piccoli che perdono contesto
    """
    sezioni = {}
    sezione_corrente = "Introduzione"
    contenuto_corrente = []
    
    linee = testo.split('\n')
    
    for linea in linee:
        linea_pulita = linea.strip()
        
        # Identifica titoli di sezione (linee corte in maiuscolo o con keywords)
        if (len(linea_pulita) < 50 and 
            (linea_pulita.isupper() or 
             any(kw in linea_pulita.lower() for kw in ['accesso', 'menu', 'utenti', 'pagamenti', 'prenotazioni', 'moduli', 'comunicazioni']))):
            
            # Salva sezione precedente
            if contenuto_corrente:
                sezioni[sezione_corrente] = '\n'.join(contenuto_corrente)
            
            # Inizia nuova sezione
            sezione_corrente = linea_pulita
            contenuto_corrente = []
        else:
            if linea_pulita:
                contenuto_corrente.append(linea_pulita)
    
    # Salva ultima sezione
    if contenuto_corrente:
        sezioni[sezione_corrente] = '\n'.join(contenuto_corrente)
    
    return sezioni

def genera_qa_con_grounding(chunk_testo: str, domanda_template: str, modello='llama3.1:8b', temperature=0.2) -> Tuple[str, str]:
    """
    Genera Q&A con forte grounding sul testo per evitare allucinazioni
    """
    try:
        system_message = """Sei un assistente che crea risposte ACCURATE per un manuale tecnico.
        
REGOLE CRITICHE:
1. Usa SOLO informazioni presenti nel testo fornito
2. NON inventare nomi di pulsanti, sezioni o procedure
3. Se il testo non contiene la risposta, rispondi "Il documento non contiene questa informazione"
4. Usa la TERMINOLOGIA ESATTA del documento
5. Risposte brevi e precise (max 3-4 frasi)"""

        prompt = f"""TESTO DEL MANUALE:
{chunk_testo}

DOMANDA DELL'UTENTE: {domanda_template}

Fornisci una risposta PRECISA basata ESCLUSIVAMENTE sul testo sopra. 
Se il testo non contiene la risposta, scrivi "Il documento non fornisce questa informazione".

RISPOSTA:"""

        response = ollama.chat(
            model=modello,
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': temperature,  # Temperatura MOLTO bassa
                'top_p': 0.7,
                'top_k': 10,
                'num_predict': 300,
                'repeat_penalty': 1.2
            }
        )
        
        risposta = response['message']['content'].strip()
        
        # Filtra risposte inutili
        if any(skip in risposta.lower() for skip in [
            "il documento non", 
            "non è menzionato", 
            "non contiene",
            "non fornisce"
        ]):
            return None, None
        
        return domanda_template, risposta
        
    except Exception as e:
        print(f"Errore generazione: {e}")
        return None, None

def genera_variazioni_domanda(domanda_base: str, num_variazioni=2) -> List[str]:
    """
    Genera variazioni di una domanda per aumentare robustezza
    """
    variazioni = [domanda_base]
    
    # Variazioni semplici senza LLM (più affidabili)
    mappings = {
        "Come posso": ["In che modo posso", "Come faccio a"],
        "Dove trovo": ["Dove posso trovare", "In quale sezione trovo"],
        "Quali": ["Che tipo di", "Quali sono i"],
        "Non riesco": ["Ho problemi a", "Non funziona"],
    }
    
    for pattern, sostituzioni in mappings.items():
        if pattern in domanda_base:
            for sost in sostituzioni[:num_variazioni]:
                variazione = domanda_base.replace(pattern, sost)
                if variazione != domanda_base:
                    variazioni.append(variazione)
    
    return variazioni[:num_variazioni + 1]

def valida_risposta(risposta: str, keywords_vietate: List[str]) -> bool:
    """
    Valida che la risposta non contenga allucinazioni comuni
    """
    risposta_lower = risposta.lower()
    
    # Keywords che indicano allucinazioni
    if any(kw in risposta_lower for kw in keywords_vietate):
        return False
    
    # Risposta troppo corta o troppo generica
    if len(risposta) < 30 or risposta.startswith("Per ") and len(risposta) < 50:
        return False
    
    return True

# =============================================================================
# FUNZIONE PRINCIPALE MIGLIORATA
# =============================================================================

def crea_dataset_qualita(
    file_input: str,
    file_output: str,
    contesto_sistema: str = CONTESTO_SISTEMA,
    modello: str = 'llama3.1:8b',
    temperature: float = 0.2,
    max_esempi_per_sezione: int = 5
):
    """
    Crea dataset di alta qualità con validazione
    """
    try:
        print(f"\n{'='*70}")
        print(f"GENERAZIONE DATASET: {file_output}")
        print(f"Temperature: {temperature} | Modello: {modello}")
        print(f"{'='*70}\n")
        
        # Carica e analizza documento
        with open(file_input, 'r', encoding='utf-8') as f:
            testo_completo = f.read()
        
        sezioni = estrai_sezioni_documento(testo_completo)
        print(f"Documento diviso in {len(sezioni)} sezioni logiche")
        
        dataset = []
        esempi_generati = 0
        esempi_scartati = 0
        
        # Keywords da evitare (allucinazioni comuni)
        keywords_vietate = [
            'dolore', 'richieste', 'mezi', 'servizi',
            'prazzi', 'daviti', 'abbracci', 'consolazione',
            'database', 'attesa', 'bolletta', 'acqua'
        ]
        
        # Per ogni sezione del documento
        for sezione_nome, sezione_testo in sezioni.items():
            print(f"\nProcessando sezione: {sezione_nome}")
            
            if len(sezione_testo) < 100:
                print(f"  Sezione troppo corta, saltata")
                continue
            
            esempi_sezione = 0
            
            # Prova diverse domande template
            for domanda_base in DOMANDE_TEMPLATE:
                if esempi_sezione >= max_esempi_per_sezione:
                    break
                
                # Genera variazioni della domanda
                variazioni = genera_variazioni_domanda(domanda_base, num_variazioni=1)
                
                for domanda in variazioni:
                    if esempi_sezione >= max_esempi_per_sezione:
                        break
                    
                    # Genera risposta con grounding
                    dom, risp = genera_qa_con_grounding(
                        sezione_testo, 
                        domanda, 
                        modello=modello,
                        temperature=temperature
                    )
                    
                    if dom and risp:
                        # Valida risposta
                        if valida_risposta(risp, keywords_vietate):
                            conversazione = {
                                "conversations": [
                                    {
                                        "role": "system",
                                        "content": f"Sei un assistente specializzato nella gestione di {contesto_sistema}."
                                    },
                                    {"role": "user", "content": dom},
                                    {"role": "assistant", "content": risp}
                                ]
                            }
                            
                            dataset.append(conversazione)
                            esempi_generati += 1
                            esempi_sezione += 1
                            print(f"  ✓ Esempio {esempi_sezione}: {dom[:50]}...")
                        else:
                            esempi_scartati += 1
                            print(f"  ✗ Risposta scartata (validazione fallita)")
                    
                    time.sleep(0.5)  # Pausa breve
        
        # Salva dataset
        with open(file_output, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"RISULTATI:")
        print(f"  Esempi generati: {esempi_generati}")
        print(f"  Esempi scartati: {esempi_scartati}")
        print(f"  Tasso successo: {esempi_generati/(esempi_generati+esempi_scartati)*100:.1f}%")
        print(f"  File salvato: {file_output}")
        print(f"{'='*70}\n")
        
        return dataset
        
    except Exception as e:
        print(f"Errore critico: {e}")
        return None

# =============================================================================
# ESECUZIONE MULTI-CONFIGURAZIONE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATORE DATASET MULTI-CONFIGURAZIONE")
    print("="*70)
    
    configurazioni = [
        {
            "nome": "ultra_conservativo",
            "temperature": 0.1,
            "descrizione": "Massima precisione, minime allucinazioni"
        },
        {
            "nome": "conservativo", 
            "temperature": 0.2,
            "descrizione": "Alta precisione, risposte accurate"
        },
        {
            "nome": "bilanciato",
            "temperature": 0.4,
            "descrizione": "Bilanciamento precisione/varietà"
        },
        {
            "nome": "esplorativo",
            "temperature": 0.6,
            "descrizione": "Maggiore varietà nelle risposte"
        }
    ]
    
    for config in configurazioni:
        print(f"\n{'='*70}")
        print(f"CONFIG: {config['nome'].upper()}")
        print(f"Descrizione: {config['descrizione']}")
        print(f"{'='*70}")
        
        crea_dataset_qualita(
            file_input='test.txt',
            file_output=f"dataset_{config['nome']}.json",
            temperature=config['temperature'],
            max_esempi_per_sezione=5
        )
        
        print(f"\nPausa tra configurazioni...")
        time.sleep(3)
    
    print("\n" + "="*70)
    print("GENERAZIONE COMPLETATA!")
    print("="*70)
    print("\nFile generati:")
    for config in configurazioni:
        print(f"  - dataset_{config['nome']}.json")
    print("\nProssimo step: Esegui training con i nuovi dataset")
