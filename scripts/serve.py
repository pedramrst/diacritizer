"""
Lightweight local web app to test the diacritizer interactively: a textarea
where you type raw Persian text and get the diacritized result back, without
re-invoking the CLI for every sentence. Loads the model once at startup.

No new dependencies -- built on http.server (stdlib) rather than
Flask/FastAPI, since this is a solo dev tool, not a deployed service.

Usage:
    python scripts/serve.py
    python scripts/serve.py --model ./canine-fa-diacritizer --port 8080
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from dotenv import load_dotenv
from transformers import CanineTokenizer, CanineForTokenClassification

from diacritizer.preprocessing import render_with_labels

load_dotenv()

DEFAULT_MODEL = "PedramR/canine-fa-diacritizer"

PAGE = """<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<title>Diacritizer -- interactive test</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 20px; }
  textarea { width: 100%; height: 120px; font-size: 20px; font-family: inherit;
             padding: 10px; box-sizing: border-box; }
  button { margin-top: 10px; padding: 8px 20px; font-size: 15px; cursor: pointer; }
  #output { margin-top: 20px; padding: 14px; min-height: 60px; font-size: 22px;
            border: 1px solid #8884; border-radius: 8px; white-space: pre-wrap; }
  #status { color: #888; font-size: 13px; margin-top: 6px; }
</style>
</head>
<body>
  <h1>Persian Diacritizer -- interactive test</h1>
  <p id="model-name" style="color:#888; font-size:13px;"></p>
  <textarea id="input" placeholder="متن فارسی بدون اعراب را اینجا وارد کنید..."></textarea>
  <br>
  <button id="run">Diacritize</button>
  <span id="status"></span>
  <div id="output" dir="rtl"></div>

<script>
  const input = document.getElementById("input");
  const output = document.getElementById("output");
  const status = document.getElementById("status");
  const btn = document.getElementById("run");

  fetch("/model").then(r => r.json()).then(d => {
    document.getElementById("model-name").textContent = "Model: " + d.model;
  });

  async function run() {
    const text = input.value.trim();
    if (!text) return;
    btn.disabled = true;
    status.textContent = "running...";
    output.textContent = "";
    try {
      const res = await fetch("/diacritize", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text})
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      output.textContent = data.result;
      status.textContent = "done in " + data.ms + " ms";
    } catch (e) {
      status.textContent = "error";
      output.textContent = String(e);
    } finally {
      btn.disabled = false;
    }
  }

  btn.addEventListener("click", run);
  input.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
  });
</script>
</body>
</html>
"""


def diacritize(text, model, tokenizer, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=2048).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    preds = torch.argmax(logits, dim=-1)[0].tolist()
    char_preds = preds[1:1 + len(text)]
    return render_with_labels(text, char_preds)


def make_handler(model, tokenizer, device, model_name):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep stdout quiet -- this is an interactive local tool

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/model":
                self._send_json({"model": model_name})
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/diacritize":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                text = payload["text"]
            except (json.JSONDecodeError, KeyError):
                self._send_json({"error": "expected JSON body {\"text\": \"...\"}"}, 400)
                return

            import time
            start = time.time()
            result = diacritize(text, model, tokenizer, device)
            elapsed_ms = round((time.time() - start) * 1000, 1)
            self._send_json({"result": result, "ms": elapsed_ms})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="local checkpoint dir, or an HF Hub repo id "
                         f"(default: {DEFAULT_MODEL})")
    ap.add_argument("--subfolder", default="",
                    help="subfolder within --model holding config.json/model.safetensors, "
                         "if any -- the Hub repo is a standard single-model repo (model "
                         "files at the root) by default, so this is normally left empty")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hub_token = os.environ.get("HF_MODEL_TOKEN")

    print(f"Loading {args.model} (subfolder={args.subfolder!r}) ...")
    tokenizer = CanineTokenizer.from_pretrained(
        args.model, subfolder=args.subfolder, token=hub_token
    )
    model = CanineForTokenClassification.from_pretrained(
        args.model, subfolder=args.subfolder, token=hub_token
    ).to(device).eval()
    print(f"Loaded on {device}.")

    handler = make_handler(model, tokenizer, device, args.model)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving at {url} -- Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
