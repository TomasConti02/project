from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import socket

app = FastAPI()

# Monta la cartella static per servire CSS, JS, immagini
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    # Legge il file HTML
    with open("templates/index.html", "r", encoding="utf-8") as file:
        return file.read()

@app.post("/echo", response_class=HTMLResponse)
def echo(text_input: str = Form(...)):
    HOST = '127.0.0.1'
    PORT = 65432
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(text_input.encode())
        data = s.recv(1024)
    
    return data.decode()  # Riceve già il testo in maiuscolo

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
