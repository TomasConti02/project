# rag_ollama_chatbot.py - Chatbot naturale e conversazionale
from sentence_transformers import SentenceTransformer
import faiss, re
import requests
import json
import time

print("🤖 INIZIALIZZAZIONE CHATBOT NATURALE...")

# ==================== CONFIGURAZIONE ====================
DOCUMENT_PATH = "test.txt"
OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api/generate"

# ==================== SISTEMA RAG OTTIMIZZATO ====================
class ChatbotRAGSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.index = None
        self.index_loaded = False
    
    def load_document(self, file_path):
        """Carica e processa il documento"""
        try:
            print(f"📖 Caricamento documento: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            cleaned = self.clean_text(text)
            self.chunks = self.create_chunks(cleaned)
            
            print(f"📄 Testo pulito: {len(cleaned)} caratteri")
            print(f"✂️ Chunk creati: {len(self.chunks)}")
            
            # Crea embeddings e indice
            print("🧠 Creazione embeddings...")
            embeddings = self.model.encode(self.chunks, normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings.astype("float32"))
            self.index_loaded = True
            
            print(f"✅ RAG PRONTO! {len(self.chunks)} chunk indicizzati")
            return True
            
        except Exception as e:
            print(f"❌ Errore nel caricamento: {e}")
            return False
    
    def clean_text(self, text):
        text = re.sub(r'[•·]', '•', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def create_chunks(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        
        for sentence in sentences:
            if len(current + sentence) > 250 and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current += " " + sentence if current else sentence
        if current:
            chunks.append(current.strip())
        return chunks
    
    def get_chatbot_context(self, query, k=3):
        """Ricerca ottimizzata per chatbot"""
        if not self.index_loaded:
            return ""
        
        emb = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(emb.astype("float32"), k)
        
        relevant_chunks = []
        for s, i in zip(scores[0], idxs[0]):
            if s > 0.3:  # Soglia bilanciata per precisione
                relevant_chunks.append(self.chunks[i])
        
        return " ".join(relevant_chunks) if relevant_chunks else ""

# ==================== CHATBOT NATURALE ====================
class NaturalChatbot:
    def __init__(self, model="mistral"):
        self.model = model
        self.base_url = "http://localhost:11434"
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.conversation_history = []
    
    def check_connection(self):
        """Verifica connessione a Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def natural_chat(self, message, context=""):
        """Chat naturale e conversazionale"""
        # Prepara il contesto della conversazione
        messages = []
        
        # System prompt per un chatbot naturale
        system_prompt = """Sei un assistente amichevole e utile del portale eCivis. 
Rispondi in modo NATURALE e CONVERSAZIONALE, come se stessi parlando con un utente.
Usa un tono amichevole ma professionale.
Se hai informazioni dal contesto, usale per rispondere in modo preciso.
Se non hai informazioni sufficienti, sii onesto ma mantieni un tono utile.
Usa frasi brevi e un linguaggio semplice."""

        messages.append({"role": "system", "content": system_prompt})
        
        # Aggiungi history della conversazione (ultime 4 scambi)
        messages.extend(self.conversation_history[-8:])
        
        # Costruisci il messaggio finale con contesto se disponibile
        user_message = message
        if context:
            user_message = f"Contesto sul portale eCivis: {context}\n\nDomanda dell'utente: {message}"
        
        messages.append({"role": "user", "content": user_message})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,  # Più creatività per conversazioni naturali
                "top_p": 0.9,
                "num_predict": 400,  # Risposte concise ma complete
                "repeat_penalty": 1.1
            }
        }
        
        try:
            response = requests.post(self.chat_url, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()
            bot_response = result["message"]["content"]
            
            # Aggiorna la storia della conversazione
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": bot_response})
            
            return bot_response
            
        except Exception as e:
            return f"Mi dispiace, ho avuto un problema tecnico. Potresti ripetere? ({e})"
    
    def get_quick_response(self, query, context=""):
        """Risposta rapida per domande semplici"""
        quick_responses = {
            'ciao': 'Ciao! 😊 Come posso aiutarti con il portale eCivis oggi?',
            'salve': 'Salve! Sono qui per assisterti con il portale eCivis.',
            'buongiorno': 'Buongiorno! 🌞 Come posso esserti utile?',
            'buonasera': 'Buonasera! Sono a disposizione per informazioni sul portale.',
            'grazie': 'Di nulla! Se hai altre domande, sono qui! 👍',
            'grazie mille': 'Figurati! Felice di esserti stato utile! 😊',
            'arrivederci': 'Arrivederci! 👋 Torna pure quando vuoi!',
            'aiuto': 'Certamente! Posso aiutarti con: accesso al portale, pagamenti, prenotazioni, moduli e molto altro. Cosa ti serve?'
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
        print("🔧 Configurazione chatbot...")
        
        if not self.rag.load_document(DOCUMENT_PATH):
            return False
        
        if not self.chatbot.check_connection():
            print("❌ Impossibile connettersi a Ollama")
            return False
        
        self.setup_complete = True
        print("✅ CHATBOT PRONTO!")
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
                return f"Piacere di conoscerti, {self.user_name}! 😊 Come posso aiutarti con il portale eCivis?"
        
        # 3. Cerca contesto rilevante
        context = self.rag.get_chatbot_context(user_input)
        
        # 4. Genera risposta naturale
        response = self.chatbot.natural_chat(user_input, context)
        
        return response

# ==================== INTERFACCIA CHATBOT NATURALE ====================
def natural_chat_interface():
    """Interfaccia chatbot naturale e user-friendly"""
    system = ChatbotSystem()
    
    if not system.setup():
        return
    
    print("\n" + "="*50)
    print("🤖 ASSISTENTE VIRTUALE eCivis")
    print("="*50)
    print("Ciao! Sono il tuo assistente per il portale eCivis.")
    print("Posso aiutarti con:")
    print("  • Accesso e login al portale")
    print("  • Pagamenti e ricariche") 
    print("  • Prenotazioni mensa")
    print("  • Moduli online")
    print("  • E molto altro!")
    print("\nScrivi pure la tua domanda...")
    print("="*50)
    
    message_count = 0
    
    while True:
        try:
            user_input = input("\n👤 Tu: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['/exit', 'esci', 'arrivederci', 'ciao']:
                farewell = "Arrivederci! 👋 Torna pure quando hai bisogno di aiuto!"
                if system.user_name:
                    farewell = f"Arrivederci, {system.user_name}! 👋 Alla prossima!"
                print(f"\n🤖 {farewell}")
                break
            
            if user_input.lower() in ['/clear', 'reset']:
                system.chatbot.clear_history()
                print("🤖 💬 Conversazione resettata! Ricominciamo da capo.")
                continue
            
            if user_input.lower() == '/help':
                print("\n🤖 💡 **Come posso aiutarti:**")
                print("• 'Come accedo al portale?'")
                print("• 'Quali metodi di pagamento ci sono?'")
                print("• 'Come disdire un pasto?'") 
                print("• 'Posso modificare un modulo inviato?'")
                print("• 'Mi chiamo Marco' → Personalizza le risposte")
                print("• '/clear' → Resetta la conversazione")
                print("• 'Grazie' → Risposta cortese")
                continue
            
            # Processa il messaggio
            message_count += 1
            if message_count == 1:
                print("🤖 ", end="", flush=True)
            else:
                print("\n🤖 ", end="", flush=True)
            
            start_time = time.time()
            response = system.process_message(user_input)
            response_time = time.time() - start_time
            
            print(response)
            
            # Mostra tempo di risposta solo se > 2 secondi
            if response_time > 2:
                print(f"   ⏱️  ({response_time:.1f}s)")
            
        except KeyboardInterrupt:
            print("\n\n🤖 Arrivederci! Spero di esserti stato utile! 👋")
            break
        except Exception as e:
            print(f"\n🤖 ❌ Ops, qualcosa è andato storto. Riprova! ({e})")

# ==================== DEMO CONVERSAZIONALE ====================
def demo_conversation():
    """Mostra una demo di conversazione naturale"""
    print("\n" + "🎭 DEMO CONVERSAZIONALE")
    print("="*50)
    
    demo_scenarios = [
        ("Utente", "Ciao!"),
        ("Bot", "Ciao! 😊 Come posso aiutarti con il portale eCivis oggi?"),
        ("", ""),
        ("Utente", "Mi chiamo Marco"),
        ("Bot", "Piacere di conoscerti, Marco! 😊 Come posso aiutarti con il portale eCivis?"),
        ("", ""),
        ("Utente", "Come faccio ad accedere al portale?"),
        ("Bot", "Per accedere al portale eCivis, devi collegarti a https://comune.ecivis.it/ e fare il login con le tue credenziali SPID o CIE. È piuttosto semplice!"),
        ("", ""),
        ("Utente", "E se voglio disdire un pasto?"),
        ("Bot", "Puoi disdire il pasto direttamente dalla sezione Prenotazioni! Seleziona l'utente, il servizio refezione e clicca sul giorno per segnare l'assenza. Il sistema salva automaticamente."),
        ("", ""),
        ("Utente", "Grazie mille!"),
        ("Bot", "Figurati, Marco! 😊 Se hai altre domande, sono qui!")
    ]
    
    for speaker, text in demo_scenarios:
        if speaker == "Utente":
            print(f"👤 {speaker}: {text}")
        elif speaker == "Bot":
            print(f"🤖 {speaker}: {text}")
        else:
            print()
    
    print("\n" + "="*50)
    input("Premi Enter per provare il chatbot...")

# ==================== ESECUZIONE PRINCIPALE ====================
def main():
    print("🤖 ASSISTENTE VIRTUALE eCivis")
    print("1. Inizia chat")
    print("2. Vedi demo")
    print("3. Esci")
    
    while True:
        scelta = input("\nScegli (1-3): ").strip()
        
        if scelta == "1":
            natural_chat_interface()
            break
        elif scelta == "2":
            demo_conversation()
            natural_chat_interface()
            break
        elif scelta == "3":
            print("Arrivederci! 👋")
            break
        else:
            print("❌ Scelta non valida")

if __name__ == "__main__":
    main()
