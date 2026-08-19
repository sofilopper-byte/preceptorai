import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


APP_DIR = Path(__file__).parent
ALUMNOS_PATH = APP_DIR / "alumnos.json"
MODEL = "gemini-3.5-flash"

with open(ALUMNOS_PATH, encoding="utf-8") as f:
    ALUMNOS = json.load(f)

client = genai.Client()  
@retry(
    retry=retry_if_exception_type(genai_errors.ServerError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30)
)
def llamar_gemini(**kwargs):
    return client.models.generate_content(**kwargs)
app = FastAPI(title="PreceptorAI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



def consultar_alumnos(cursos: list[str]) -> str:
    """Devuelve alumnos y sus tutores para una lista de cursos.

    Args:
        cursos: Nombres de curso a buscar, ej. ["Primero Primera", "Cuarto Segunda"].
                Si se necesita TODA la escuela, pasar ["TODOS"].

    Returns:
        JSON string con la lista de alumnos/tutores encontrados, agrupados por curso.
    """
    if any(c.strip().upper() == "TODOS" for c in cursos):
        seleccion = ALUMNOS
    else:
        cursos_norm = {c.strip().lower() for c in cursos}
        seleccion = [a for a in ALUMNOS if a["curso"].strip().lower() in cursos_norm]

    agrupado: dict[str, list[dict]] = {}
    for alumno in seleccion:
        agrupado.setdefault(alumno["curso"], []).append(alumno)

    return json.dumps(agrupado, ensure_ascii=False)


EXTRACTION_PROMPT = """Sos un asistente que interpreta avisos escolares desestructurados
escritos por un preceptor. Extraé la información en JSON con este esquema exacto:

{{
  "reglas": [
    {{
      "cursos": ["nombre de curso tal cual se usa en la base, ej 'Cuarto Primera'"],
      "situacion": "resumen corto de qué cambia para este grupo (horario, no asiste, sancion, salida, etc)",
      "motivo": "motivo mencionado, si lo hay"
    }}
  ],
  "fecha_mencionada": "fecha o dia mencionado en el aviso, si lo hay, si no null"
}}

Si el aviso menciona "todos" o "el resto del colegio", usa el curso especial "TODOS".
Si el aviso usa apodos de curso (ej "4to y 5to"), expandilos a los nombres reales de curso
suponiendo el formato "<Numero en palabras> <Primera/Segunda>" (ej "4to" -> "Cuarto Primera"
y "Cuarto Segunda" si no se especifica division, incluí ambas).

Aviso del preceptor:
\"\"\"{aviso}\"\"\"

Devolvé SOLO el JSON, sin texto adicional ni markdown.
"""


def extraer_reglas(aviso: str) -> dict:
    try:
        resp = llamar_gemini(
            model=MODEL,
            contents=EXTRACTION_PROMPT.format(aviso=aviso),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
    except genai_errors.ServerError:
        raise HTTPException(status_code=503, detail="Gemini está saturado, reintentá en unos segundos.")
    return json.loads(resp.text)



GENERATION_PROMPT = """Sos PreceptorAI, un asistente que redacta comunicados escolares
formales para familias, en español rioplatense pero formal (de "usted").

Reglas detectadas en el aviso original del preceptor:
{reglas_json}

Para CADA curso involucrado, primero llamá a la herramienta consultar_alumnos para
saber qué alumnos y tutores están afectados. Después redactá UN comunicado por curso,
dirigido a "Familias de <curso>", breve (3-5 lineas), formal, claro, sin inventar
datos que no estén en las reglas.

Devolvé el resultado final como JSON con este esquema, sin texto adicional ni markdown:
{{
  "comunicados": [
    {{
      "curso": "...",
      "cantidad_familias": 0,
      "texto": "..."
    }}
  ]
}}
"""


def generar_comunicados(reglas: dict) -> dict:
    try:
        resp = llamar_gemini(
            model=MODEL,
            contents=GENERATION_PROMPT.format(
                reglas_json=json.dumps(reglas, ensure_ascii=False)
            ),
            config=types.GenerateContentConfig(
                tools=[consultar_alumnos],
                temperature=0.3,
            ),
        )
    except genai_errors.ServerError:
        raise HTTPException(status_code=503, detail="Gemini está saturado, reintentá en unos segundos.")

    texto = resp.text.strip()
    if texto.startswith("```"):
        texto = texto.strip("`")
        texto = texto.split("\n", 1)[1] if "\n" in texto else texto
        if texto.lower().startswith("json"):
            texto = texto[4:]
    return json.loads(texto)

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EnvioCursoIn(BaseModel):
    curso: str
    texto: str


def enviar_email(destinatarios: list[str], asunto: str, cuerpo: str) -> None:
    email_user = os.environ["EMAIL_USER"]
    email_password = os.environ["EMAIL_PASSWORD"]

    msg = MIMEMultipart()
    msg["From"] = email_user
    msg["To"] = email_user  # el "To" es la cuenta propia, los tutores van en CC
    msg["Cc"] = ", ".join(destinatarios)
    msg["Subject"] = asunto
    msg.attach(MIMEText(cuerpo, "plain"))

    todos_destinatarios = [email_user] + destinatarios

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(email_user, email_password)
        server.sendmail(email_user, todos_destinatarios, msg.as_string())


@app.post("/enviar-curso")
def enviar_curso(body: EnvioCursoIn):
    curso_norm = body.curso.strip().lower()
    alumnos_curso = [a for a in ALUMNOS if a["curso"].strip().lower() == curso_norm]

    if not alumnos_curso:
        raise HTTPException(status_code=404, detail=f"No se encontraron alumnos para el curso '{body.curso}'.")

    emails_tutores = sorted({a["email"] for a in alumnos_curso if a.get("email")})

    if not emails_tutores:
        raise HTTPException(status_code=404, detail=f"No hay emails de tutores cargados para '{body.curso}'.")

    try:
        enviar_email(
            destinatarios=emails_tutores,
            asunto=f"Comunicado - {body.curso}",
            cuerpo=body.texto,
        )
    except smtplib.SMTPException as e:
        raise HTTPException(status_code=502, detail=f"Error al enviar el email: {e}")

    return {"enviado_a": emails_tutores, "cantidad": len(emails_tutores)}


# API

class AvisoIn(BaseModel):
    aviso: str


@app.post("/procesar")
def procesar(body: AvisoIn):
    reglas = extraer_reglas(body.aviso)
    comunicados = generar_comunicados(reglas)
    return {
        "reglas_detectadas": reglas,
        "comunicados": comunicados["comunicados"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "alumnos_cargados": len(ALUMNOS)}


app.mount("/", StaticFiles(directory=str(APP_DIR / "static"), html=True), name="static")