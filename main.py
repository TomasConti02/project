from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
#FastAPI → il framework web.
#Request → oggetto che rappresenta la richiesta HTTP.
app = FastAPI() #creo istanza 

# cartelle statiche e template
app.mount("/static", StaticFiles(directory="static"), name="static") #indico dove si trovano i file css statici
templates = Jinja2Templates(directory="templates") #indico file html della page 

#Quando visiti http://localhost:8000/ con un GET, viene caricata la pagina index.html.
# pagina iniziale con il form
@app.get("/", response_class=HTMLResponse)
def home(request: Request): #all'arrivo della get su 8000/ viene eseguita operazione di home
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# ricezione della stringa dal form
@app.post("/echo", response_class=HTMLResponse)
def echo_string(request: Request, text: str = Form(...)):
    return templates.TemplateResponse("index.html", {"request": request, "result": text})
