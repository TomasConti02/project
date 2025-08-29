import socket
import ollama
from threading import Thread

HOST = '127.0.0.1'
PORT = 65432

class OllamaWorker:
    def __init__(self):
        self.history = {}
    
    def get_user_history(self, addr):
        if addr not in self.history:
            self.history[addr] = []
        return self.history[addr]

    def generate_response(self, user_input, history, addr):
        try:
            # Mantieni solo le ultime 6 interazioni per non gonfiare troppo il contesto
            messages = history[-6:] if history else []

            # Inseriamo sempre il system message come primo elemento
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {
                    "role": "system", 
                    "content": "Rispondi SOLO con le informazioni del tuning. Non inventare categorie o dettagli che non esistono."
                })

            # Aggiungiamo l'input dell'utente
            messages.append({"role": "user", "content": user_input})

            # Generazione con parametri controllati
            response = ollama.chat(
                model="ModelTest:latest",
                messages=messages,
                options={
                    "temperature": 0.0,   # massimo determinismo
                    "num_predict": 300,   # limite lunghezza
                    "top_p": 0.9          # riduce esplorazione
                }
            )

            ai_response = response['message']['content'].strip()

            # Aggiorniamo la cronologia dell'utente
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": ai_response})
            
            return ai_response

        except Exception as e:
            return f"Errore: {str(e)}"

def handle_client(conn, addr, worker):
    history = worker.get_user_history(addr)
    
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break

            user_input = data.decode('utf-8').strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                conn.sendall("Arrivederci!\n".encode('utf-8'))
                break

            ai_response = worker.generate_response(user_input, history, addr)
            conn.sendall((ai_response + "\n").encode('utf-8'))

    finally:
        conn.close()

def start_server():
    worker = OllamaWorker()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server in ascolto su {HOST}:{PORT}")
        
        while True:
            conn, addr = s.accept()
            client_thread = Thread(target=handle_client, args=(conn, addr, worker))
            client_thread.daemon = True
            client_thread.start()

if __name__ == "__main__":
    start_server()
