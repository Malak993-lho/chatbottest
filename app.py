# app.py
import os
import re
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any

# ====== Settings ======
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
BASE_DIR = Path(__file__).parent
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gemma2:2b")

app = FastAPI()

# Serve index.html at /
@app.get("/", response_class=HTMLResponse)
def home():
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>TamTam API</h1>"

# Serve /static if exists
if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ====== Lang detection ======
_AR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_EN = re.compile(r"[A-Za-z]")

def detect_lang(text: str) -> str:
    if _AR.search(text):
        return "ar"
    if _EN.search(text):
        return "en"
    return "ar"  # default Arabic

# ====== System prompt ======
SYSTEM_PROMPT = """
You are TamTam, a child-safe, friendly assistant for kids and families.
Keep answers brief (≤ 6 lines), warm, supportive, and age-appropriate.
Do NOT give medical or legal advice. If pain, risk, self-harm, abuse, or danger is mentioned:
- Encourage telling a trusted adult NOW and contacting local emergency services.
- Protect privacy (no full names, addresses, phone numbers, IDs, or passwords).

Language rules:
- If [lang=ar], reply in Arabic ONLY. Do NOT translate to English, do NOT mix languages.
- Use simple, friendly Arabic. Add a gentle emoji or two when appropriate.
- Be gentle, curious, and kind. Prefer short sentences and questions that invite sharing.
""".strip()

# ====== Memory (per-session in-process LRU) ======
# Simple in-memory store: { session_id: {"name": "Malak", ...} }
# NOTE: For production, replace with Redis/DB and rotate/expire safely.
class LRUStore(OrderedDict):
    def __init__(self, capacity=1000):
        super().__init__()
        self.capacity = capacity
    def get_mem(self, sid: str) -> Dict[str, Any]:
        if not sid:
            return {}
        if sid in self:
            self.move_to_end(sid)
            return super().__getitem__(sid)
        mem = {}
        super().__setitem__(sid, mem)
        if len(self) > self.capacity:
            self.popitem(last=False)
        return mem
    def set_mem(self, sid: str, mem: Dict[str, Any]):
        if not sid:
            return
        super().__setitem__(sid, mem)
        self.move_to_end(sid)
        if len(self) > self.capacity:
            self.popitem(last=False)

MEM = LRUStore(capacity=2000)

# Helpers to read/write memory safely
def mem_get(req_body: dict) -> Dict[str, Any]:
    sid = (req_body.get("session_id") or "").strip()
    return MEM.get_mem(sid) if sid else {}
def mem_set(req_body: dict, mem: Dict[str, Any]):
    sid = (req_body.get("session_id") or "").strip()
    if sid:
        MEM.set_mem(sid, mem)

# ====== Name extraction patterns ======
# Arabic examples: "اسمي ملك", "انا اسمي ملك", "اسمي هو ملك", "انا ملك"
# English examples: "my name is Malak", "I'm Malak", "I am Malak"
NAME_AR_PAT = re.compile(
    r"(?:\bاسمي(?:\s*هو)?\s+|(?:انا|أنا)\s+(?:اسمي\s+)?)"
    r"([A-Za-z\u0600-\u06FF][A-Za-z\u0600-\u06FF\s]{0,30})"
)
NAME_EN_PAT = re.compile(
    r"(?:\bmy\s+name\s+is\s+|(?:i\s*am|i'm)\s+)"
    r"([A-Za-z][A-Za-z\s'-]{0,30})",
    re.IGNORECASE,
)
ASK_NAME_AR_PAT = re.compile(r"(شو\s+اسمي|شو\s+اسمي؟|بتعرفني|بتعرف\s+اسمي|بتعرفني؟)")
ASK_NAME_EN_PAT = re.compile(r"\b(what(?:'s|\s+is)\s+my\s+name|do\s+you\s+know\s+me)\b", re.IGNORECASE)

def clean_name(raw: str) -> str:
    name = raw.strip()
    # remove trailing punctuation/emojis common
    name = re.sub(r"[\s\.,!?\u200f\u200e]+$", "", name)
    # collapse spaces
    name = re.sub(r"\s{2,}", " ", name)
    # limit length
    return name[:40]

# ====== Rules (simple guardrails) ======
RULES = []

def rule(pattern):
    rx = re.compile(pattern, re.IGNORECASE)
    def _wrap(fn):
        RULES.append((rx, fn))
        return fn
    return _wrap

# --- Self-harm: fixed Arabic reply (and English fallback) ---
SELF_HARM_RX = r"(?:\b(?:suicide|kill(?:\s*myself)?|end\s*my\s*life|i\s*want\s*to\s*die|i'?m\s*going\s*to\s*kill\s*myself|self[-\s]?harm)\b|انتح(?:ر|ار)|\bبدي\s*انتحر\b|رح\s*انتحر|أ(?:ن|ح)تحر|اقتل\s*نفسي|قتل\s*حالي|أنهي\s*حياتي)"
@rule(SELF_HARM_RX)
def on_self_harm(text: str, lang: str):
    if lang == "ar":
        return {"reply": (
            "شو عم بصير؟ خبرني شو صار؟ كيف فيني ساعدك؟ 💛🙏\n"
            "إذا في خطر فوري عليك أو حاسس/ة إنك ممكن تأذي حالك، احكي فورًا مع شخص بالغ موثوق أو اتصل/ي بخدمات الطوارئ."
        )}
    else:
        return {"reply": (
            "What’s happening? Tell me what happened and how I can help. 💛🙏\n"
            "If there’s immediate danger, please reach out to a trusted adult or contact emergency services."
        )}

# --- Sadness/anxiety handler (short) ---
@rule(r"\b(scared|afraid|anxious|nervous|worried)\b|(\bخا(?:يف|يفة)|قلقان|قلقانة|مرعوب|مرعوبة|متوتر|متوترة|قَلِق|زعلان|زعل|حزين|حزينة)\b")
def on_anxiety(text: str, lang: str):
    if lang == "ar":
        return {"reply": "ليش زعلان؟ خبرني شو صار وكيف فيني ساعدك؟ 💛"}
    return {"reply": "Why are you feeling down? Tell me what happened and how I can help. 💛"}

# --- Wait / pause ---
@rule(r"\b(wait|later|not now|hold on)\b|(?:(?:مش|مو)\s*هلق)|\b(بعدين|انطر|نطر|خليني|خلّيني|خليني شوي)\b")
def on_wait(text: str, lang: str):
    if lang == "ar":
        return {"reply": "أكيد. خبرني قديش بدّك ننطر: 5 أو 10 أو 15 دقيقة؟ ولما تجهّز/ي قُل/ي «جاهز». 🙂"}
    return {"reply": "Sure. How long should we wait: 5, 10, or 15 minutes? When you’re ready, say “ready”. 🙂"}

# --- Risk / abuse ---
@rule(r"\b(hurt|abuse|bully|kill|die)\b|(?:اضربوني|يؤذيني|عنف|خطر|مهدد|تنمّر|تنمر)")
def on_risk(text: str, lang: str):
    if lang == "ar":
        return {"reply": "إذا في خطر عليك أو على غيرك، إحكي فورًا مع شخص بالغ موثوق. وإذا عاجل، اتصل/ي بالطوارئ حالًا. إذا بتحب/ي تحكي أكتر، أنا عم بسمعك. 🙏"}
    return {"reply": "If you or someone else is in danger, tell a trusted adult now. If it’s urgent, call emergency services. If you want to share more, I’m listening. 🙏"}

# --- Minute picker ---
@rule(r"\b(5|10|15)\s*(min|mins|minutes)?\b|(?:\b5\b|\b10\b|\b15\b)\s*(?:د(?:ق(?:يقة|ايق)|قيقة)|دقايق)?")
def on_minutes(text: str, lang: str):
    m = re.search(r"(5|10|15)", text)
    mins = m.group(1) if m else "5"
    if lang == "ar":
        return {"reply": f"تمام. لما تجهّز/ي بعد {mins} دقيقة، قولي «جاهز». ملاحظة: ما في منبّه حقيقي هون. 🙂"}
    return {"reply": f"Okay. When you’re ready in {mins} minutes, say “ready”. Note: there’s no real timer here. 🙂"}

@rule(r"\b(ready|i'm ready|جاهز|جاهزة)\b")
def on_ready(text: str, lang: str):
    if lang == "ar":
        return {"reply": "يا سلام! جاهزين نكمل 💪 شو بتحب/ي نحكي هلق؟"}
    return {"reply": "Awesome! Let’s continue 💪 What would you like to talk about now?"}

def apply_rules(text: str, lang: str):
    for rx, fn in RULES:
        if rx.search(text):
            return fn(text, lang)
    return None

# ====== Chat endpoint ======
@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    user_msg = (body.get("prompt") or body.get("message") or "").strip()
    if not user_msg:
        return JSONResponse({"error": "empty message"}, status_code=400)

    model = body.get("model", DEFAULT_MODEL)
    req_lang = (body.get("lang") or "").lower()
    lang = req_lang if req_lang in {"ar", "en"} else detect_lang(user_msg)

    # ----- Memory: load (based on session_id) -----
    mem = mem_get(body)

    # ----- 0) Name capture / recall (BEFORE rules & LLM) -----
    # Capture name
    name = None
    if lang == "ar":
        m = NAME_AR_PAT.search(user_msg)
        if m:
            name = clean_name(m.group(1))
    else:
        m = NAME_EN_PAT.search(user_msg)
        if m:
            name = clean_name(m.group(1))

    if name:
        mem["name"] = name
        mem_set(body, mem)
        if lang == "ar":
            return {"reply": f"تشرفت يا {name}! رح أتذكّر اسمك. 🙂 شو بتحب/ي نحكي؟", "lang": lang, "model": model, "rule_applied": True}
        else:
            return {"reply": f"Nice to meet you, {name}! I’ll remember your name. 🙂 What would you like to talk about?", "lang": lang, "model": model, "rule_applied": True}

    # Answer questions like "what is my name" / "do you know me"
    if (lang == "ar" and ASK_NAME_AR_PAT.search(user_msg)) or (lang != "ar" and ASK_NAME_EN_PAT.search(user_msg)):
        stored = mem.get("name")
        if stored:
            if lang == "ar":
                return {"reply": f"اسمك {stored} 🙂", "lang": lang, "model": model, "rule_applied": True}
            else:
                return {"reply": f"Your name is {stored} 🙂", "lang": lang, "model": model, "rule_applied": True}
        else:
            if lang == "ar":
                return {"reply": "لسّه ما بعرف اسمك. بتحب/ي تقلّي: «اسمي …» لأتذكّرو؟ 🙂", "lang": lang, "model": model, "rule_applied": True}
            else:
                return {"reply": "I don’t know your name yet. You can tell me: “my name is …” so I remember. 🙂", "lang": lang, "model": model, "rule_applied": True}

    # ----- 1) Try rules (self-harm, etc.) -----
    rule_hit = apply_rules(user_msg, lang)
    if rule_hit:
        return {"reply": rule_hit["reply"], "lang": lang, "model": model, "rule_applied": True}

    # ----- 2) Call LLM (inject memory context) -----
    history = body.get("history") or []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add memory context (non-identifying; name only if user gave it)
    if mem.get("name"):
        if lang == "ar":
            messages.append({"role": "system", "content": f"User profile: الاسم={mem['name']}. إذا [lang=ar] الجواب بالعربي فقط."})
        else:
            messages.append({"role": "system", "content": f"User profile: name={mem['name']}."})

    for m in history:
        if isinstance(m, dict) and "role" in m and "content" in m:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": f"[lang={lang}] {user_msg}"})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.6, "top_p": 0.9, "num_ctx": 2048},
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            r.raise_for_status()
            out = r.json()

            reply = ""
            if isinstance(out, dict):
                reply = out.get("message", {}).get("content", "") or out.get("response", "") or out.get("result", "")
            if not isinstance(reply, str):
                reply = str(reply)
            reply = reply.strip()

            # Arabic-only safety filter
            if lang == "ar":
                latin_letters = sum(1 for c in reply if c.isascii() and c.isalpha())
                if latin_letters > 0.25 * max(len(reply), 1) or len(reply.splitlines()) > 6:
                    reply = "خلّينا نكمّل بالعربي بس. كيف فيني ساعدك؟ 🙂"

            return {"reply": reply, "lang": lang, "model": model, "rule_applied": False}
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"Ollama error: {e}"}, status_code=502)
