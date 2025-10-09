"""
Spieghiamo il RAG
RAG = Retrieval-Augmented Generation e lo dividiamo in 2 fasi:
1. Retrieval (recupero): trova i testi più rilevanti nel tuo database o documenti (es. manuali e FAQ in test.json).
2. Generation (generazione): passa quei testi al modello LLM (qui llama3.2:1b via Ollama) per creare una risposta naturale e coerente.
EnhancedRAGSystem gestisce il recupero dei documenti che sarebbero dei json ben strutturati. Ciascuno di essi viene suddiviso in chuncks
e convertito in embedding vettoriale. Al ternine memorizzato dentro un indice vettoriale FAISS.
BISONA MIGLIORARE IL RAG come ?
suggerimenti:
1. metadati dentro i documenti json, da dare al LLM come contesto che non li sto usando
2. allungare chuck a 400-600 parole per prendere più frasi per embedding
3. Aggiungi un piccolo "overlap" tra chunk per evitare tagli improvvisi di frasi
4. La soglia fissa s > 0.25 va bene, ma puoi migliorarla:
   Alza se i risultati sono troppi o poco precisi.
   Abbassala se il contesto è vuoto troppo spesso.
   Puoi anche renderla dinamica in base al tipo di query:
        threshold = 0.25 if query_analysis['complexity'] == 'high' else 0.35
//-----------------------------------------------------------------------------//
//-----------------------------------------------------------------------------//
Altro tipo di embadding
{
  "documents": [
    {
      "id": "doc_001",
      "title": "Accesso al portale eCivis",
      "category": "Accesso e autenticazione",
      "content": "Per accedere al portale eCivisWeb è necessario utilizzare SPID o CIE...",
      "keywords": ["login", "SPID", "CIE", "autenticazione", "portale"],
      "questions": [
        "Come accedo al portale eCivis?",
        "Posso entrare con la carta d'identità elettronica?"
      ]
    }
  ],
  "faq": [
    {
      "id": "faq_001",
      "category": "Pagamenti",
      "question": "Come posso ricaricare il credito mensa?",
      "answer": "Puoi ricaricare il credito mensa accedendo alla sezione Pagamenti dal portale..."
    }
  ]
}

"""
from sentence_transformers import SentenceTransformer
import faiss, re
import requests
import json
import time

print("🤖 INIZIALIZZAZIONE CHATBOT MIGLIORATO...")

# ==================== CONFIGURAZIONE ====================
DOCUMENT_PATH = "test.json"
OLLAMA_MODEL = "llama3.2:1b"

# ==================== SISTEMA DI ADATTAMENTO MIGLIORATO ====================
class EnhancedQueryAnalyzer:
    def __init__(self):
        self.general_questions = {
            'chi_sei': ['chi sei', 'cosa sei', 'presentati', 'chi mi parla'],
            'aiuto': ['cosa puoi fare', 'cosa sai fare', 'come puoi aiutarmi', 'funzionalità'],
            'saluti': ['ciao', 'buongiorno', 'buonasera', 'salve', 'hey'],
            'grazie': ['grazie', 'grazie mille', 'ti ringrazio'],
            'addio': ['arrivederci', 'esco', 'exit', 'quit'],
            'stato': ['come stai', 'tutto bene', 'come va']
        }
        
        self.complex_keywords = {
            'high': ['spiegami', 'descrivimi', 'guida completa', 'procedura dettagliata', 'tutorial', 'passo dopo passo'],
            'medium': ['come fare', 'istruzioni', 'procedura', 'modalità', 'passaggi', 'funziona'],
            'low': ['cos\'è', 'che cosa', 'definizione', 'dove', 'quando', 'quanto', 'quale']
        }
    
    def analyze_query(self, query):
        query_lower = query.lower()
        
        # Controlla domande generali PRIMA
        for category, keywords in self.general_questions.items():
            if any(keyword in query_lower for keyword in keywords):
                return {
                    'type': 'general',
                    'category': category,
                    'needs_context': False,
                    'complexity': 'low',
                    'response_length': 150
                }
        
        # Stima complessità
        complexity = 'low'
        for level, keywords in self.complex_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                complexity = level
                break
        
        # Determina lunghezza
        length_map = {'low': 250, 'medium': 350, 'high': 450}
        
        return {
            'type': 'specific',
            'category': 'eCivis',
            'needs_context': True,
            'complexity': complexity,
            'response_length': length_map[complexity]
        }

# ==================== SISTEMA RAG MIGLIORATO ====================
class EnhancedRAGSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        self.index_loaded = False

    def load_document(self, file_path):
        try:
            print(f"📖 Caricamento documento JSON: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            all_contents = []
            
            for doc in data.get("documents", []):
                content = self.clean_text(doc["content"])
                # Aggiungi titolo e domande correlate per migliorare la ricerca
                enhanced_content = f"Titolo: {doc['title']}. Contenuto: {content}. Domande correlate: {' '.join(doc.get('questions', []))}"
                all_contents.append({
                    "content": enhanced_content,
                    "metadata": {
                        "id": doc["id"],
                        "category": doc["category"],
                        "title": doc["title"],
                        "type": "document"
                    }
                })
            
            for faq in data.get("faq", []):
                enhanced_content = f"FAQ: {faq['question']} - Risposta: {faq['answer']}"
                all_contents.append({
                    "content": enhanced_content,
                    "metadata": {
                        "id": faq["id"],
                        "category": faq["category"],
                        "type": "faq"
                    }
                })
            
            print(f"📄 Documenti caricati: {len(data.get('documents', []))}")
            print(f"❓ FAQ caricate: {len(data.get('faq', []))}")
            
            self.chunks = []
            self.chunk_metadata = []
            
            for item in all_contents:
                content_chunks = self.create_chunks(item["content"])
                for chunk in content_chunks:
                    self.chunks.append(chunk)
                    self.chunk_metadata.append(item["metadata"])
            
            print(f"✂️ Chunk totali creati: {len(self.chunks)}")

            embeddings = self.model.encode(self.chunks, normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings.astype("float32"))
            self.index_loaded = True

            print(f"✅ RAG PRONTO! {len(self.chunks)} chunk indicizzati")
            return True

        except Exception as e:
            print(f"❌ Errore nel caricamento JSON: {e}")
            return False

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def create_chunks(self, text, max_length=350):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def get_chatbot_context(self, query, k=4):
        if not self.index_loaded:
            return ""

        emb = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(emb.astype("float32"), k)

        relevant_chunks = []
        
        for s, i in zip(scores[0], idxs[0]):
            if s > 0.25:  # Soglia più bassa per più risultati
                chunk_text = self.chunks[i]
                relevant_chunks.append(chunk_text)
        
        return " ".join(relevant_chunks) if relevant_chunks else ""

# ==================== CHATBOT MIGLIORATO ====================
class EnhancedChatbot:
    def __init__(self, model="llama3.2:1b"):
        self.model = model
        self.base_url = "http://localhost:11434"
        self.generate_url = f"{self.base_url}/api/generate"
        self.conversation_history = []
        self.query_analyzer = EnhancedQueryAnalyzer()

    def check_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_general_response(self, query_analysis, user_input=""):
        responses = {
            'chi_sei': "Ciao! Sono il tuo assistente virtuale specializzato sul portale eCivisWeb. 🤖 Posso aiutarti con informazioni su accesso, pagamenti, prenotazioni mensa, moduli online e tutti i servizi del portale!",
            'aiuto': "Posso aiutarti con: 🔐 Accesso (SPID/CIE), 💰 Pagamenti e ricariche, 🍽️ Prenotazioni mensa, 📝 Moduli online, 📊 Gestione utenti, 📄 Comunicazioni. Cosa ti serve?",
            'saluti': "Ciao! 😊 Sono qui per aiutarti con il portale eCivisWeb. Come posso assisterti oggi?",
            'grazie': "Di nulla! 😊 Sono felice di esserti stato utile. Se hai altre domande, sono qui!",
            'addio': "Arrivederci! 👋 Torna pure quando hai bisogno di aiuto con eCivis!",
            'stato': "Sto bene, grazie! 😊 Pronto ad aiutarti con il portale eCivisWeb. Cosa ti serve?"
        }
        
        # Se l'utente ha detto il nome, personalizza
        if "mi chiamo" in user_input.lower():
            name_match = re.search(r'mi chiamo\s+([A-Za-z]+)', user_input.lower())
            if name_match:
                name = name_match.group(1).strip().title()
                return f"Ciao {name}! 😊 Sono il tuo assistente eCivis. Come posso aiutarti?"
        
        return responses.get(query_analysis['category'], "Ciao! Come posso aiutarti con eCivis?")

    def natural_chat(self, message, context="", query_analysis=None):
        
        # Se è una domanda generale, usa risposta predefinita
        if query_analysis['type'] == 'general':
            return self.get_general_response(query_analysis, message)

        # PREPARA IL PROMPT MIGLIORATO
        system_prompt = """Sei un assistente ESPERTO del portale eCivisWeb. 
DEVI usare le informazioni dal contesto per rispondere alla domanda.
Se il contesto contiene informazioni rilevanti, fornisci una risposta UTILE e PRATICA.
Se il contesto non è sufficiente, fornisci comunque una risposta GENERICA basata sulle tue conoscenze di eCivis.
NON dire "non ho informazioni" o "non posso aiutarti".
Mantieni un tono AMICHEVOLE e PROFESSIONALE."""

        full_prompt = f"CONTESTO eCivis (usa queste informazioni):\n{context}\n\n" if context else ""
        full_prompt += f"DOMANDA UTENTE: {message}\n\n"
        full_prompt += "RISPOSTA ASSISTENTE:"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": query_analysis['response_length'],
                "repeat_penalty": 1.1
            }
        }

        try:
            print(f"🔄 Invio a Ollama ({query_analysis['complexity']} complexity)...")
            response = requests.post(self.generate_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            bot_response = result["response"].strip()

            # Pulisci la risposta
            bot_response = self.clean_response(bot_response)
            
            # Aggiorna history
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": bot_response})

            return bot_response

        except Exception as e:
            return f"Mi dispiace, al momento non posso rispondere. Riprova tra un momento."

    def clean_response(self, response):
        """Pulisce la risposta da frasi indesiderate"""
        unwanted_phrases = [
            "mi dispiace ma non ho",
            "non ho abbastanza informazioni",
            "non posso aiutarti",
            "non ho sufficienti informazioni",
            "non ho informazioni specifiche"
        ]
        
        for phrase in unwanted_phrases:
            if phrase in response.lower():
                # Sostituisci con una risposta più utile
                response = response.lower().replace(phrase, "posso dirti che")
                response = response.capitalize()
        
        return response

    def get_quick_response(self, query):
        quick_responses = {
            'ciao': 'Ciao! 😊 Sono il tuo assistente eCivis. Come posso aiutarti?',
            'grazie': 'Di nulla! Se hai altre domande, sono qui! 👍',
        }
        
        query_lower = query.lower().strip()
        for key, response in quick_responses.items():
            if key in query_lower:
                return response
        return None

    def clear_history(self):
        self.conversation_history = []

# ==================== SISTEMA PRINCIPALE MIGLIORATO ====================
class EnhancedChatbotSystem:
    def __init__(self):
        self.rag = EnhancedRAGSystem()
        self.chatbot = EnhancedChatbot()
        self.setup_complete = False
        self.user_name = None

    def setup(self):
        print("🔧 Configurazione chatbot eCivis...")
        if not self.rag.load_document(DOCUMENT_PATH):
            return False
        if not self.chatbot.check_connection():
            print("❌ Impossibile connettersi a Ollama")
            return False
        self.setup_complete = True
        print("✅ CHATBOT eCivis PRONTO!")
        return True

    def process_message(self, user_input):
        if not self.setup_complete:
            return "Sistema non pronto."

        # 1. Analizza la query
        query_analysis = self.chatbot.query_analyzer.analyze_query(user_input)
        print(f"📊 Tipo: {query_analysis['type']} | Complessità: {query_analysis['complexity']}")

        # 2. Risposte rapide
        quick_response = self.chatbot.get_quick_response(user_input)
        if quick_response:
            return quick_response

        # 3. Estrazione nome (migliorata)
        if not self.user_name and any(word in user_input.lower() for word in ['mi chiamo', 'sono', 'il mio nome è']):
            name_match = re.search(r'(?:mi chiamo|sono|il mio nome è)\s+([A-Za-zÀ-ÿ]+)', user_input, re.IGNORECASE)
            if name_match:
                self.user_name = name_match.group(1).strip().title()
                return f"Piacere di conoscerti, {self.user_name}! 😊 Come posso aiutarti con eCivis?"

        # 4. Ricerca contesto MIGLIORATA
        context = ""
        if query_analysis['needs_context']:
            context = self.rag.get_chatbot_context(user_input)
            if context:
                print(f"🔍 Contesto trovato ({len(context)} caratteri)")
            else:
                print("🔍 Ricerca contesto...")

        # 5. Genera risposta
        response = self.chatbot.natural_chat(user_input, context, query_analysis)
        
        # 6. Personalizza se abbiamo il nome
        if self.user_name and query_analysis['type'] == 'specific':
            response = f"{self.user_name}, {response.lower()}" if not response.startswith(self.user_name) else response

        return response

# ==================== INTERFACCIA MIGLIORATA ====================
def enhanced_chat_interface():
    system = EnhancedChatbotSystem()
    if not system.setup():
        return

    print("\n" + "="*50)
    print("🤖 ASSISTENTE eCivisWeb MIGLIORATO")
    print("="*50)
    print("Prova queste domande:")
    print("• 'Come accedo al portale?'")
    print("• 'Come funziona la mensa?'") 
    print("• 'Come prenoto un pasto?'")
    print("• 'Come faccio i pagamenti?'")
    print("• 'Chi è il genitore intestatario?'")
    print("="*50)

    while True:
        try:
            user_input = input("\n👤 Tu: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ['/exit', 'esci']:
                farewell = "Arrivederci! 👋"
                if system.user_name:
                    farewell = f"Arrivederci, {system.user_name}! 👋"
                print(f"\n🤖 {farewell}")
                break

            if user_input.lower() == '/clear':
                system.chatbot.clear_history()
                print("🤖 💬 Conversazione resettata!")
                continue

            if user_input.lower() == '/name':
                print(f"🤖 Nome utente: {system.user_name or 'Non impostato'}")
                continue

            print("🤖 ", end="", flush=True)
            start_time = time.time()
            response = system.process_message(user_input)
            response_time = time.time() - start_time

            print(response)
            print(f"   ⏱️  ({response_time:.1f}s)")

        except KeyboardInterrupt:
            print("\n\n🤖 Arrivederci! 👋")
            break
        except Exception as e:
            print(f"\n🤖 ❌ Errore: {e}")

if __name__ == "__main__":
    enhanced_chat_interface()
