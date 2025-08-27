from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# cartelle statiche e template
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# pagina iniziale con il form
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# ricezione della stringa dal form
@app.post("/echo", response_class=HTMLResponse)
def echo_string(request: Request, text: str = Form(...)):
    return templates.TemplateResponse("index.html", {"request": request, "result": text})
