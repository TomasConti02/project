from sentence_transformers import SentenceTransformer
import faiss, re
import requests
import json
import time

print("🤖 INIZIALIZZAZIONE CHATBOT NATURALE...")

# ==================== CONFIGURAZIONE ====================
DOCUMENT_PATH = "test.json"
OLLAMA_MODEL = "llama3.2:1b"  # Modello che hai su Colab
OLLAMA_URL = "http://localhost:11434/api/generate"

# ==================== SISTEMA RAG OTTIMIZZATO ====================
class ChatbotRAGSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        self.index_loaded = False

    def load_document(self, file_path):
        """Carica e processa il documento JSON"""
        try:
            print(f"📖 Caricamento documento JSON: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            all_contents = []
            
            # Processa documents
            for doc in data.get("documents", []):
                content = self.clean_text(doc["content"])
                enhanced_content = f"{doc['title']}. {content} Keywords: {', '.join(doc['keywords'])}"
                all_contents.append({
                    "content": enhanced_content,
                    "metadata": {
                        "id": doc["id"],
                        "category": doc["category"],
                        "title": doc["title"],
                        "type": "document"
                    }
                })
            
            # Processa FAQ
            for faq in data.get("faq", []):
                enhanced_content = f"Domanda: {faq['question']} Risposta: {faq['answer']}"
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
            
            # Crea chunks dai contenuti
            self.chunks = []
            self.chunk_metadata = []
            
            for item in all_contents:
                content_chunks = self.create_chunks(item["content"])
                for chunk in content_chunks:
                    self.chunks.append(chunk)
                    self.chunk_metadata.append(item["metadata"])
            
            print(f"✂️ Chunk totali creati: {len(self.chunks)}")

            # Crea embeddings e indice
            print("🧠 Creazione embeddings...")
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
        text = re.sub(r'[•·]', '•', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def create_chunks(self, text, max_length=300):
        """Crea chunks più intelligenti dal testo"""
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
        """Ricerca ottimizzata con metadati"""
        if not self.index_loaded:
            return ""

        emb = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(emb.astype("float32"), k)

        relevant_chunks = []
        used_categories = set()
        
        for s, i in zip(scores[0], idxs[0]):
            if s > 0.25:
                chunk_text = self.chunks[i]
                metadata = self.chunk_metadata[i]
                
                if metadata["type"] == "faq":
                    chunk_with_source = f"[FAQ - {metadata['category']}] {chunk_text}"
                else:
                    chunk_with_source = f"[{metadata['title']} - {metadata['category']}] {chunk_text}"
                
                relevant_chunks.append(chunk_with_source)
                used_categories.add(metadata["category"])
        
        if relevant_chunks:
            print(f"🔍 Categorie trovate: {', '.join(used_categories)}")
        
        return "\n".join(relevant_chunks) if relevant_chunks else ""

    def get_suggested_questions(self, query, k=3):
        """Suggerisce domande correlate basate sulla similarità"""
        if not self.index_loaded:
            return []
        
        emb = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(emb.astype("float32"), k)
        
        suggestions = []
        for s, i in zip(scores[0], idxs[0]):
            if s > 0.3 and self.chunk_metadata[i]["type"] == "document":
                metadata = self.chunk_metadata[i]
                suggestions.append(f"{metadata['title']} ({metadata['category']})")
        
        return suggestions[:3]

# ==================== CHATBOT NATURALE CORRETTO ====================
class NaturalChatbot:
    def __init__(self, model="llama3.2:1b"):  # Usa il modello che hai
        self.model = model
        self.base_url = "http://localhost:11434"
        self.generate_url = f"{self.base_url}/api/generate"  # Endpoint corretto
        self.conversation_history = []

    def check_connection(self):
        """Verifica connessione a Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def natural_chat(self, message, context=""):
        """Chat usando l'endpoint generate (corretto)"""
        # Costruisci il prompt completo per l'endpoint generate
        system_prompt = """Sei un assistente esperto del portale eCivisWeb. 
Usa le informazioni dal contesto fornito per rispondere in modo preciso.
Se non hai informazioni sufficienti nel contesto, sii onesto.
Mantieni un tono amichevole ma professionale.
Organizza le risposte in modo chiaro quando possibile."""

        # Prepara il prompt completo per l'endpoint generate
        full_prompt = f"Sistema: {system_prompt}\n\n"
        
        if context:
            full_prompt += f"CONTESTO:\n{context}\n\n"
        
        # Aggiungi history della conversazione
        if self.conversation_history:
            full_prompt += "STORIA CONVERSAZIONE:\n"
            for msg in self.conversation_history[-4:]:  # Ultimi 4 messaggi
                role = "Utente" if msg["role"] == "user" else "Assistente"
                full_prompt += f"{role}: {msg['content']}\n"
            full_prompt += "\n"
        
        full_prompt += f"Utente: {message}\n\nAssistente:"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 350,  # Lunghezza risposta
                "repeat_penalty": 1.1
            }
        }

        try:
            print("🔄 Invio a Ollama...")
            response = requests.post(self.generate_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            bot_response = result["response"].strip()

            # Aggiorna la storia della conversazione
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": bot_response})

            return bot_response

        except requests.exceptions.RequestException as e:
            print(f"❌ Errore di rete: {e}")
            return "Mi dispiace, ho problemi di connessione con il servizio. Riprova tra un momento."
        except Exception as e:
            print(f"❌ Errore generico: {e}")
            return "Mi dispiace, ho avuto un problema tecnico. Potresti ripetere?"

    def get_quick_response(self, query, context=""):
        """Risposta rapida per domande semplici"""
        quick_responses = {
            'ciao': 'Ciao! 😊 Sono il tuo assistente per il portale eCivisWeb. Come posso aiutarti?',
            'salve': 'Salve! Sono qui per assisterti con il portale eCivisWeb.',
            'buongiorno': 'Buongiorno! 🌞 Come posso esserti utile con i servizi eCivis?',
            'buonasera': 'Buonasera! Sono a disposizione per informazioni sul portale eCivis.',
            'grazie': 'Di nulla! Se hai altre domande sui servizi eCivis, sono qui! 👍',
            'grazie mille': 'Figurati! Felice di esserti stato utile! 😊',
            'arrivederci': 'Arrivederci! 👋 Torna pure quando hai bisogno di aiuto con eCivis!',
            'aiuto': 'Certamente! Posso aiutarti con: accesso al portale, pagamenti, prenotazioni mensa, moduli online e molto altro. Cosa ti serve?'
        }

        query_lower = query.lower().strip()
        for key, response in quick_responses.items():
            if key in query_lower:
                return response

        return None

    def clear_history(self):
        """Pulisce la storia della conversazione"""
        self.conversation_history = []

# ==================== SISTEMA CHATBOT PRINCIPALE ====================
class ChatbotSystem:
    def __init__(self):
        self.rag = ChatbotRAGSystem()
        self.chatbot = NaturalChatbot()
        self.setup_complete = False
        self.user_name = None

    def setup(self):
        """Setup completo del sistema"""
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
        """Elabora un messaggio in modo naturale"""
        if not self.setup_complete:
            return "Il sistema non è pronto, riprova tra un momento."

        # 1. Controlla se è una risposta rapida predefinita
        quick_response = self.chatbot.get_quick_response(user_input)
        if quick_response:
            return quick_response

        # 2. Estrai il nome se è un saluto personale
        if not self.user_name:
            name_match = re.search(r'(?:mi chiamo|sono|il mio nome è)\s+([^.?!]*)', user_input.lower())
            if name_match:
                self.user_name = name_match.group(1).strip().title()
                return f"Piacere di conoscerti, {self.user_name}! 😊 Come posso aiutarti con il portale eCivisWeb?"

        # 3. Cerca contesto rilevante
        context = self.rag.get_chatbot_context(user_input)
        
        # 4. Ottieni suggerimenti per domande correlate
        suggestions = self.rag.get_suggested_questions(user_input)

        # 5. Genera risposta naturale
        response = self.chatbot.natural_chat(user_input, context)
        
        # 6. Aggiungi suggerimenti se disponibili
        if suggestions and "non ho informazioni" not in response.lower() and "non so" not in response.lower():
            response += f"\n\n💡 **Potresti anche chiedere:**\n" + "\n".join([f"• {s}" for s in suggestions])

        return response

# ==================== INTERFACCIA CHATBOT NATURALE ====================
def natural_chat_interface():
    """Interfaccia chatbot naturale e user-friendly"""
    system = ChatbotSystem()

    if not system.setup():
        print("❌ Impossibile avviare il chatbot. Controlla la connessione Ollama.")
        return

    print("\n" + "="*50)
    print("🤖 ASSISTENTE VIRTUALE eCivisWeb")
    print("="*50)
    print("Ciao! Sono il tuo assistente specializzato sul portale eCivisWeb.")
    print("Posso aiutarti con:")
    print("  • Accesso e autenticazione (SPID/CIE)")
    print("  • Gestione utenti e nucleo familiare")
    print("  • Pagamenti e ricariche mensa")
    print("  • Prenotazioni e disdette pasti")
    print("  • Moduli online e bandi")
    print("  • Comunicazioni e documenti")
    print("\nScrivi pure la tua domanda...")
    print("="*50)

    message_count = 0

    while True:
        try:
            user_input = input("\n👤 Tu: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['/exit', 'esci', 'arrivederci', 'ciao']:
                farewell = "Arrivederci! 👋 Torna pure quando hai bisogno di aiuto con eCivis!"
                if system.user_name:
                    farewell = f"Arrivederci, {system.user_name}! 👋 Alla prossima!"
                print(f"\n🤖 {farewell}")
                break

            if user_input.lower() in ['/clear', 'reset']:
                system.chatbot.clear_history()
                print("🤖 💬 Conversazione resettata! Ricominciamo da capo.")
                continue

            if user_input.lower() == '/help':
                print("\n🤖 💡 **Come posso aiutarti con eCivis:**")
                print("• 'Come accedo con SPID?'")
                print("• 'Come ricarico il conto mensa?'")
                print("• 'Come disdico un pasto?'")
                print("• 'Dove trovo le comunicazioni?'")
                print("• 'Come presento un modulo online?'")
                print("• 'Mi chiamo Marco' → Personalizza le risposte")
                print("• '/clear' → Resetta la conversazione")
                print("• 'Grazie' → Risposta cortese")
                print("• '/stats' → Mostra statistiche")
                continue

            if user_input.lower() == '/stats':
                print(f"\n📊 **Statistiche:**")
                print(f"• Nome utente: {system.user_name or 'Non specificato'}")
                print(f"• Messaggi in history: {len(system.chatbot.conversation_history)}")
                print(f"• Chunk nel sistema: {len(system.rag.chunks)}")
                continue

            # Processa il messaggio
            message_count += 1
            print("🤖 ", end="", flush=True)

            start_time = time.time()
            response = system.process_message(user_input)
            response_time = time.time() - start_time

            print(response)

            # Mostra tempo di risposta
            print(f"   ⏱️  ({response_time:.1f}s)")

        except KeyboardInterrupt:
            print("\n\n🤖 Arrivederci! Spero di esserti stato utile con eCivis! 👋")
            break
        except Exception as e:
            print(f"\n🤖 ❌ Ops, qualcosa è andato storto: {e}")

# ==================== ESECUZIONE PRINCIPALE ====================
if __name__ == "__main__":
    natural_chat_interface()
