# Chatbot Demo - Servizio Clienti
Questa repository contiene un **chatbot ottimizzato (tuned)** per la gestione del servizio clienti.  
Si tratta di una **demo di test e prova**.

---
## Dataset
Il dataset utilizzato per il tuning del modello è disponibile su Hugging Face:
[ProjectDataset.json](https://huggingface.co/datasets/tomasconti/TestTuning/blob/main/ProjectDataset.json)
---
## Requisiti
Assicurarsi di avere installato:

- [Ollama Server](https://ollama.com/download/)
- Python 3.x
- GPU Nvidia + CUDA or Google colab Tesla T4 runtime
---
## Requisiti
esecuzione codice TuningCode.py per adestrare il modello 
scaricare la dir di salvataggio

## Setup del modello Ollama
1. All'interno della dir di addestramento
2. Creare il modello Ollama con il comando:  
   ```bash
   ollama create ModelTest -f Modelfile
  
---
## Avvio chatbot
 cd socket
 python3 socket.py
---
## Avvio web server
uvicorn main:app --reload
Aprire il browser su: http://127.0.0.1:8000















////COSE DI TOMAS :)
# nuova-repo
#pip install fastapi uvicorn jinja2

launch -> uvicorn main:app --reload
server open on http://127.0.0.1:8000
python3 echo_server.py
aprire porta per parlare telnet 127.0.0.1 12345
pip install torch unsloth transformers accelerate
ls ~/.cache/huggingface/hub
