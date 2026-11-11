"""Gradio front end. It is a pure API client: everything goes through the FastAPI service (no agent code here).

Accounts (task 5): register / log in on the right; the token travels as `Authorization: Bearer <token>`. Until your API has
POST /users the UI works anonymously, so you always have something to click."""
import os
import uuid

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage pings to Hugging Face from the students' UI

import gradio as gr
import httpx

API = os.getenv("API_URL", "http://localhost:8000")                # server-side calls
PUBLIC_API = os.getenv("PUBLIC_API_URL", "http://localhost:8000")   # URLs the browser opens (chart iframes)
http = httpx.Client(base_url=API, timeout=300)  # ingestion = OCR + several Groq calls


def detail(r: httpx.Response) -> str:
    """Readable error text from any failed API response (JSON detail, validation list, or a plain-text body)."""
    try:
        d = r.json().get("detail", r.text)
    except ValueError:
        d = r.text
    if isinstance(d, list):  # FastAPI validation errors
        d = "; ".join(str(e.get("msg", e)) for e in d)
    return str(d)[:300] or f"HTTP {r.status_code}"


def auth(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"} if tok else {}


LOGIN_HINT = "_Log in (or register) on the right to use your knowledge base._"


def new_sid() -> str:
    return uuid.uuid4().hex[:12]


def kb_markdown(sid: str, tok: str = "") -> str:
    r = http.get(f"/kb/{sid}", headers=auth(tok))
    if r.status_code == 401:
        return LOGIN_HINT
    kb = r.json()
    if not kb["documents"]:
        return "_Empty. Upload a PDF, DOCX, TXT or image on the left._"
    lines = [f"**Overview:** {kb['overview']}", f"**Keywords:** {', '.join(kb['keywords'])}", ""]
    for d in kb["documents"]:
        ocr = f", {d['ocr_pages']} OCR pages" if d["ocr_pages"] else ""
        lines.append(f"- **{d['filename']}** ({d['doc_type']}; {d['n_sections']} sections, {d['n_chunks']} chunks{ocr})  \n  {d['summary']}")
    return "\n".join(lines)


def chart_html(urls: list[str]) -> str:
    if not urls:
        return "<i>No chart yet. Ask for one, e.g. top 5 artists in Chinook as a bar chart.</i>"
    return f'<iframe src="{PUBLIC_API}{urls[-1]}" style="width:100%;height:480px;border:0"></iframe>'


def ask(message: str, sid: str, tok: str = "") -> dict:
    try:
        r = http.post("/agent", json={"message": message, "thread_id": sid}, headers=auth(tok))
    except httpx.HTTPError:
        return {"answer": "Error: the API is not reachable. Please try again in a moment.", "steps": [], "charts": []}
    if r.status_code == 401:
        return {"answer": "Please log in first (account box on the right).", "steps": [], "charts": []}
    if r.status_code == 403:
        return {"answer": "This chat belongs to another user. Click 'New session'.", "steps": [], "charts": []}
    if r.status_code != 200:
        return {"answer": f"Error {r.status_code}: {detail(r)}", "steps": [], "charts": []}
    return r.json()


def turn(message, history, sid, tok=""):
    """One chat turn: show the user message at once, then the answer, steps, chart and knowledge base."""
    history = [*history, {"role": "user", "content": message}, {"role": "assistant", "content": "Working..."}]
    yield history, gr.update(), gr.update(), gr.update(), ""
    res = ask(message, sid, tok)
    history[-1] = {"role": "assistant", "content": res["answer"]}
    yield history, res["steps"], chart_html(res["charts"]), kb_markdown(sid, tok), ""


def upload(paths, history, sid, tok=""):
    """Upload each file to the API, then tell the agent about it. The agent decides to call ingest_document."""
    final = (history, gr.update(), gr.update(), gr.update(), None)
    for path in paths or []:
        try:
            with open(path, "rb") as f:
                r = http.post("/upload", data={"session_id": sid}, files={"file": (os.path.basename(path), f)}, headers=auth(tok))
        except httpx.HTTPError:
            history = [*history, {"role": "assistant", "content": "Upload failed: the API is not reachable."}]
            final = (history, *final[1:])
            continue
        if r.status_code != 200:
            why = "please log in first" if r.status_code == 401 else detail(r)
            history = [*history, {"role": "assistant", "content": f"Upload failed: {why}"}]
            final = (history, *final[1:])
            continue
        up = r.json()
        msg = f"I uploaded '{up['filename']}' (file_id={up['file_id']}). Please add it to the knowledge base."
        for out in turn(msg, history, sid, tok):
            history = out[0]
            final = (*out[:4], None)  # 5th output here is the File component (turn's 5th is the textbox: "" would resolve to a path)
            yield final
    yield final  # always end with a state, so failures (and empty uploads) show up


def reset_session(tok):
    sid = new_sid()
    return sid, [], [], chart_html([]), kb_markdown(sid, tok)


def clear_kb(sid, tok):
    http.delete(f"/kb/{sid}", headers=auth(tok))
    return kb_markdown(sid, tok)


def account(action: str, username: str, password: str, tok: str, sid: str):
    """register | login | logout -> (token, status text, new session id, chat, knowledge base)."""
    username = (username or "").strip().lower()
    try:
        if action == "logout":
            http.post("/logout", headers=auth(tok))
            return "", "Logged out.", new_sid(), [], LOGIN_HINT
        if action == "register":
            r = http.post("/users", json={"username": username, "password": password})
            if r.status_code == 404:
                return tok, "The API has no user management yet (task 5): you are anonymous.", sid, gr.update(), gr.update()
            if r.status_code != 201:
                return tok, f"Register failed: {detail(r)}", sid, gr.update(), gr.update()
        r = http.post("/login", json={"username": username, "password": password})
    except httpx.HTTPError:
        return tok, "The API is not reachable.", sid, gr.update(), gr.update()
    if r.status_code != 200:
        return tok, f"Login failed: {detail(r)}", sid, gr.update(), gr.update()
    new_tok = r.json()["token"]
    return new_tok, f"Logged in as **{username}**. Your documents are shared by all your chats.", new_sid(), [], kb_markdown(new_sid(), new_tok)


with gr.Blocks(title="Document Agent") as demo:
    sid = gr.State(new_sid)  # a CALLABLE: runs per browser session (a value would be created once and shared by everyone)
    tok = gr.State("")  # the bearer token of this browser session
    gr.Markdown("## Document Agent\nSession 1 tools (SQL, charts, OpenML, web) + your documents. Upload a file, then ask about it.")
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=480, label="Chat")
            box = gr.Textbox(placeholder="Ask anything, e.g. 'What does the contract say about termination?'", show_label=False)
            files = gr.File(label="Upload documents (PDF, DOCX, TXT, MD, images)", file_count="multiple", type="filepath")
        with gr.Column(scale=2):
            with gr.Accordion("Account", open=True):
                who = gr.Markdown("_Not logged in._")
                with gr.Row():
                    user_box = gr.Textbox(label="Username", scale=2)
                    pass_box = gr.Textbox(label="Password", type="password", scale=2)
                with gr.Row():
                    register_btn = gr.Button("Register")
                    login_btn = gr.Button("Log in", variant="primary")
                    logout_btn = gr.Button("Log out")
            with gr.Tab("Knowledge base"):
                kb = gr.Markdown("_Empty. Upload a PDF, DOCX, TXT or image on the left._")
                with gr.Row():
                    new = gr.Button("New session")
                    clear = gr.Button("Clear knowledge base")
            with gr.Tab("Steps"):
                steps = gr.JSON(label="Tool calls this turn")
            with gr.Tab("Chart"):
                chart = gr.HTML(chart_html([]))

    outputs = [chat, steps, chart, kb, box]
    demo.load(kb_markdown, [sid, tok], kb)  # fresh browser session -> its knowledge base (or the log-in hint)
    box.submit(turn, [box, chat, sid, tok], outputs)
    files.upload(upload, [files, chat, sid, tok], [chat, steps, chart, kb, files])
    new.click(reset_session, tok, [sid, chat, steps, chart, kb])
    clear.click(clear_kb, [sid, tok], kb)
    acct_out = [tok, who, sid, chat, kb]
    register_btn.click(lambda u, p, t, s: account("register", u, p, t, s), [user_box, pass_box, tok, sid], acct_out)
    login_btn.click(lambda u, p, t, s: account("login", u, p, t, s), [user_box, pass_box, tok, sid], acct_out)
    logout_btn.click(lambda u, p, t, s: account("logout", u, p, t, s), [user_box, pass_box, tok, sid], acct_out)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
