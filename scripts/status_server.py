"""Mini dashboard LAN : suivi LIVE des bancs depuis n'importe quel appareil du réseau.

Lance un serveur HTTP (stdlib, lecture seule) qui rend une page auto-rafraîchie (5 s)
avec l'état de chaque banc : ✅ terminé (tableau final) / ⏳ en cours (dernières lignes,
fraîcheur) / 🕓 en file / ⚠️ interrompu. Aucun impact sur le pipeline (lecture de fichiers).

Usage : python scripts/status_server.py   ->  http://<ip-du-mac>:8765
"""

from __future__ import annotations

import html
import http.server
import os
import subprocess
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8765

BENCHES = [
    ("A · Boucle LLML (petit doc, vs 14B)", "logs/benchmark_llml_loop.log", "benchmark_llml_loop.py"),
    ("B · MoE v2 auto-améliorant", "logs/moe_v2.log", "moe_v2.py"),
    ("C · Code skills v1 (plafonné)", "logs/benchmark_code_skills.log", "benchmark_code_skills.py"),
    ("D · Cahier 20k + débordement 32k vs 14B", "logs/benchmark_bigctx.log", "benchmark_bigctx_14b.py"),
    ("E · Experts sous charge 20k", "logs/benchmark_underload.log", "benchmark_experts_underload.py"),
    ("C2 · Code skills DURS (STaR/distill)", "logs/benchmark_code_skills2.log", "benchmark_code_skills2.py"),
    ("D2 · Legacy non-conforme + LoRA-14B", "logs/benchmark_bigctx2.log", "benchmark_bigctx2.py"),
]

CSS = """
body{background:#0d1117;color:#c9d1d9;font-family:ui-monospace,Menlo,monospace;
     margin:0;padding:14px;font-size:13px}
h1{font-size:16px;color:#e6edf3;margin:0 0 4px}
.sub{color:#8b949e;font-size:11px;margin-bottom:14px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 12px;margin-bottom:10px}
.card h2{font-size:13px;margin:0 0 6px;color:#e6edf3}
pre{margin:6px 0 0;white-space:pre-wrap;word-break:break-word;font-size:11px;line-height:1.45;
    color:#9da7b3;max-height:260px;overflow-y:auto}
.done{border-left:4px solid #2ea043}.run{border-left:4px solid #d29922}
.queued{border-left:4px solid #30363d}.dead{border-left:4px solid #f85149}
.meta{color:#8b949e;font-size:11px}
.res{color:#7ee787}
"""


def _proc_running(script):
    try:
        r = subprocess.run(["pgrep", "-f", f"scripts/{script}$"], capture_output=True, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return False


def _bench_html(title, logrel, script):
    path = os.path.join(_PROJ, logrel)
    if not os.path.isfile(path):
        return f'<div class="card queued"><h2>🕓 {title}</h2><div class="meta">en file d\'attente</div></div>'
    try:
        content = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        content = ""
    age = int(time.time() - os.path.getmtime(path))
    running = _proc_running(script)
    done = "=== FIN ===" in content
    if done:
        i = content.rfind("=== RÉSULTAT")
        block = content[i:] if i >= 0 else "\n".join(content.splitlines()[-14:])
        body = html.escape("\n".join(block.splitlines()[:24]))
        return (f'<div class="card done"><h2>✅ {title}</h2>'
                f'<pre class="res">{body}</pre></div>')
    tail = html.escape("\n".join(content.splitlines()[-12:]))
    if running:
        fresh = "🟢" if age < 300 else "🟠 (silence long — grosse génération probable)"
        return (f'<div class="card run"><h2>⏳ {title}</h2>'
                f'<div class="meta">dernière écriture il y a {age}s {fresh}</div><pre>{tail}</pre></div>')
    return (f'<div class="card dead"><h2>⚠️ {title}</h2>'
            f'<div class="meta">process absent sans FIN (interrompu ?)</div><pre>{tail}</pre></div>')


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        cards = "\n".join(_bench_html(*b) for b in BENCHES)
        page = (f'<!doctype html><html><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<meta http-equiv="refresh" content="5"><title>LLML — pipeline</title>'
                f'<style>{CSS}</style></head><body>'
                f'<h1>LLML — pipeline de benchmarks</h1>'
                f'<div class="sub">rafraîchi toutes les 5 s · {time.strftime("%H:%M:%S")} · lecture seule</div>'
                f'{cards}</body></html>')
        data = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
