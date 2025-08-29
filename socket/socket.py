import socket
import ollama
import time
from threading import Thread
import hashlib

HOST = '127.0.0.1'
PORT = 65432
MAX_HISTORY = 6        # Numero massimo di coppie domanda-risposta da mantenere
MAX_TOKENS = 256       # Numero massimo di token per risposta
TIMEOUT = 60           # Timeout più generoso
CACHE_SIZE = 1000      # Dimensione massima della cache

class OllamaWorker:
    def __init__(self):
        self.history = {}  # {addr: [messages]}
        self.response_cache = {}  # Cache per risposte duplicate
    
    def get_user_history(self, addr):
        """Ottiene o crea la history per un utente specifico"""
        if addr not in self.history:
            self.history[addr] = []
        return self.history[addr]

    def generate_cache_key(self, user_input, addr):
        """Genera una chiave unica per la cache"""
        # Normalizza l'input per evitare duplicati dovuti a spazi/case
        normalized_input = user_input.lower().strip()
        return f"{addr}_{hashlib.md5(normalized_input.encode()).hexdigest()}"

    def generate_response(self, user_input, history, addr):
        """Genera risposta usando il modello con parametri ottimizzati"""
        try:
            # Controlla cache per domande identiche
            cache_key = self.generate_cache_key(user_input, addr)
            if cache_key in self.response_cache:
                print(f"[{addr}] Utilizzando risposta dalla cache")
                return self.response_cache[cache_key]
            
            # Sistema prompt per consistenza e precisione
            system_prompt = """Sei un assistente AI utile, preciso e consistente. 
            Fornisci risposte fattuali, concise e coerenti. 
            Se non conosci la risposta, ammettilo onestamente invece di inventare.
            Mantieni un tono professionale e appropriato al contesto."""

            # Prepara i messaggi con system prompt e history
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history[-(MAX_HISTORY*2):] if history else [])
            messages.append({"role": "user", "content": user_input})

            print(f"[{addr}] Inviando {len(messages)} messaggi al modello...")
            start_time = time.time()

            # PARAMETRI OTTIMIZZATI PER CONSISTENZA
            response = ollama.chat(
                model="ModelTest:latest",
                messages=messages,
                options={
                    'num_predict': MAX_TOKENS,
                    'temperature': 0.2,           # Bassa ma non zero per flessibilità
                    'top_p': 0.5,                # Bilancia creatività e consistenza
                    'top_k': 40,                 # Limita le scelte migliori
                    'repeat_penalty': 1.15,      # Leggermente aumentato per evitare ripetizioni
                    'seed': 42,                  # Seed fisso per riproducibilità
                    'mirostat': 2,               # Modalità mirostat avanzata
                    'mirostat_tau': 2.5,         # Controllo della perplessità target
                    'mirostat_eta': 0.08,        # Tasso di apprendimento
                    'num_ctx': 2048,             # Contesto sufficiente
                    'num_thread': 4              # Ottimizzazione prestazioni
                }
            )

            processing_time = time.time() - start_time
            print(f"[{addr}] Risposta generata in {processing_time:.2f}s")

            ai_response = response['message']['content'].strip()

            # Aggiorna history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": ai_response})
            self.history[addr] = history[-(MAX_HISTORY*2):]  # Mantieni solo le ultime N coppie

            # Salva in cache (solo per risposte di successo)
            self.response_cache[cache_key] = ai_response
            
            # Limita dimensione cache con LRU-like approach
            if len(self.response_cache) > CACHE_SIZE:
                # Rimuove il primo elemento (approccio semplice)
                oldest_key = next(iter(self.response_cache))
                del self.response_cache[oldest_key]

            return ai_response

        except Exception as e:
            error_msg = f"Errore Ollama: {str(e)}"
            print(f"[{addr}] {error_msg}")
            return error_msg

def handle_client(conn, addr, worker):
    print(f"[{addr}] Connessione accettata")
    history = worker.get_user_history(addr)
    
    try:
        conn.settimeout(TIMEOUT)
        while True:
            try:
                data = conn.recv(4096)
                if not data:
                    break

                user_input = data.decode('utf-8').strip()
                print(f"[{addr}] Ricevuto: {user_input}")

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    conn.sendall("Arrivederci!\n".encode('utf-8'))
                    break

                # Genera risposta
                ai_response = worker.generate_response(user_input, history, addr)
                conn.sendall((ai_response + "\n").encode('utf-8'))

            except socket.timeout:
                conn.sendall("Timeout: risposta troppo lunga\n".encode('utf-8'))
                break
            except ConnectionResetError:
                break

    except Exception as e:
        print(f"[{addr}] Errore: {e}")
    finally:
        conn.close()
        print(f"[{addr}] Connessione chiusa")

def start_server():
    worker = OllamaWorker()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server Ollama ottimizzato in ascolto su {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            client_thread = Thread(target=handle_client, args=(conn, addr, worker))
            client_thread.daemon = True
            client_thread.start()

if __name__ == "__main__":
    start_server()
