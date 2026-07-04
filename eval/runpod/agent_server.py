#!/usr/bin/env python3
"""Minimal exec server for driving a RunPod pod over the HTTPS proxy.

Runs on the pod (stdlib only). All requests require header X-Auth == $EXEC_TOKEN.
Endpoints:
  GET  /health                    -> {ok, gpu}
  POST /exec   {cmd, cwd?}        -> {job}       (async; output -> /workspace/.jobs/<job>.log)
  GET  /poll?job=&offset=         -> {done, rc, output, size}
  POST /cancel {job}              -> {ok}
  POST /write  {path, b64, mode?} -> {ok, bytes} (append with mode="a")
  GET  /read?path=&offset=&limit= -> {b64, size, eof}
  GET  /ls?path=                  -> {entries: [{name, size, dir}]}
"""
import base64, json, os, signal, subprocess, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = os.environ.get("EXEC_TOKEN", "")
JOBS = "/workspace/.jobs"
os.makedirs(JOBS, exist_ok=True)
MAX_CHUNK = 800_000  # bytes of log per poll (stay well under proxy limits)


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        if self.headers.get("X-Auth") != TOKEN or not TOKEN:
            self._send(403, {"error": "forbidden"})
            return False
        return True

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if not self._auth():
            return
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/health":
            self._send(200, {"ok": True, "gpu": gpu_info()})
        elif u.path == "/poll":
            job = os.path.basename(q.get("job", ""))
            off = int(q.get("offset", 0))
            log, rcf = f"{JOBS}/{job}.log", f"{JOBS}/{job}.rc"
            if not os.path.exists(log):
                return self._send(404, {"error": "no such job"})
            size = os.path.getsize(log)
            with open(log, "rb") as f:
                f.seek(off)
                out = f.read(MAX_CHUNK)
            rc = None
            if os.path.exists(rcf):
                with open(rcf) as f:
                    txt = f.read().strip()
                    rc = int(txt) if txt else None
            self._send(200, {"done": rc is not None, "rc": rc, "size": size,
                             "output": out.decode("utf-8", "replace")})
        elif u.path == "/read":
            p = q.get("path", "")
            off, lim = int(q.get("offset", 0)), int(q.get("limit", 2_000_000))
            if not os.path.isfile(p):
                return self._send(404, {"error": "no such file"})
            size = os.path.getsize(p)
            with open(p, "rb") as f:
                f.seek(off)
                data = f.read(min(lim, MAX_CHUNK * 2))
            self._send(200, {"b64": base64.b64encode(data).decode(),
                             "size": size, "eof": off + len(data) >= size})
        elif u.path == "/ls":
            p = q.get("path", "/workspace")
            if not os.path.isdir(p):
                return self._send(404, {"error": "no such dir"})
            ents = []
            for name in sorted(os.listdir(p))[:500]:
                fp = os.path.join(p, name)
                ents.append({"name": name, "dir": os.path.isdir(fp),
                             "size": os.path.getsize(fp) if os.path.isfile(fp) else 0})
            self._send(200, {"entries": ents})
        else:
            self._send(404, {"error": "unknown route"})

    def do_POST(self):
        if not self._auth():
            return
        try:
            b = self._body()
        except Exception as e:
            return self._send(400, {"error": str(e)})
        if self.path == "/exec":
            job = uuid.uuid4().hex[:12]
            log, rcf = f"{JOBS}/{job}.log", f"{JOBS}/{job}.rc"
            cwd = b.get("cwd") or "/workspace"
            wrapped = f"({b['cmd']}) >> {log} 2>&1; echo $? > {rcf}"
            proc = subprocess.Popen(["bash", "-lc", wrapped], cwd=cwd,
                                    start_new_session=True)
            with open(f"{JOBS}/{job}.pid", "w") as f:
                f.write(str(proc.pid))
            open(log, "a").close()
            self._send(200, {"job": job})
        elif self.path == "/cancel":
            job = os.path.basename(b.get("job", ""))
            try:
                with open(f"{JOBS}/{job}.pid") as f:
                    pid = int(f.read())
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                with open(f"{JOBS}/{job}.rc", "w") as f:
                    f.write("137")
                self._send(200, {"ok": True})
            except Exception as e:
                self._send(500, {"error": str(e)})
        elif self.path == "/write":
            p = b["path"]
            os.makedirs(os.path.dirname(p) or "/", exist_ok=True)
            data = base64.b64decode(b["b64"])
            with open(p, "ab" if b.get("mode") == "a" else "wb") as f:
                f.write(data)
            if b.get("chmod"):
                os.chmod(p, int(str(b["chmod"]), 8))
            self._send(200, {"ok": True, "bytes": len(data)})
        else:
            self._send(404, {"error": "unknown route"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("EXEC_PORT", "8888"))
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
