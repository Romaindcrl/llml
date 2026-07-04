#!/usr/bin/env python3
"""Client for the pod's agent_server (over the RunPod HTTPS proxy).

Usage:
  python remote.py run "nvidia-smi"                 # exec + stream until done
  python remote.py run --cwd /workspace/llml "..."  # with working dir
  python remote.py bg  "long command"               # start, print job id, return
  python remote.py poll JOB [--offset N]
  python remote.py upload LOCAL REMOTE
  python remote.py download REMOTE LOCAL
  python remote.py health

Reads pod url + token from eval/runpod/.pod.json.
"""
import argparse, base64, json, os, sys, time
import urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 4_000_000  # b64 chars per upload request


def _state():
    with open(os.path.join(HERE, ".pod.json")) as f:
        return json.load(f)


def _req(method, path, body=None, params=None, timeout=90):
    st = _state()
    url = st["url"] + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"X-Auth": st["token"],
                                          "User-Agent": "curl/8.5.0",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def health():
    return _req("GET", "/health")


def exec_bg(cmd, cwd=None):
    return _req("POST", "/exec", {"cmd": cmd, "cwd": cwd})["job"]


def poll(job, offset=0):
    return _req("GET", "/poll", params={"job": job, "offset": offset})


def run(cmd, cwd=None, poll_every=5, timeout=7200, quiet=False, tail_to=None):
    """Exec and stream output until completion. Returns (rc, full_output)."""
    job = exec_bg(cmd, cwd)
    off, out, t0 = 0, [], time.time()
    while True:
        try:
            r = poll(job, off)
        except Exception as e:
            if time.time() - t0 > timeout:
                raise
            if not quiet:
                print(f"[poll retry: {e}]", file=sys.stderr)
            time.sleep(min(30, poll_every * 3))
            continue
        if r["output"]:
            out.append(r["output"])
            if not quiet:
                sys.stdout.write(r["output"])
                sys.stdout.flush()
            if tail_to:
                with open(tail_to, "a") as f:
                    f.write(r["output"])
            off += len(r["output"].encode("utf-8", "replace"))
            # server offsets are in bytes; recompute from reported size when drifting
            if off > r["size"]:
                off = r["size"]
        if r["done"] and off >= r["size"]:
            return r["rc"], "".join(out)
        if time.time() - t0 > timeout:
            raise TimeoutError(f"job {job} exceeded {timeout}s (still running on pod)")
        time.sleep(poll_every)


def upload(local, remote, chmod=None):
    size = os.path.getsize(local)
    sent, mode = 0, "w"
    with open(local, "rb") as f:
        while True:
            chunk = f.read(CHUNK // 2)
            if not chunk:
                break
            body = {"path": remote, "b64": base64.b64encode(chunk).decode(), "mode": mode}
            if chmod and mode == "w":
                body["chmod"] = chmod
            _req("POST", "/write", body, timeout=180)
            sent += len(chunk)
            mode = "a"
    return sent


def download(remote, local):
    off = 0
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    with open(local, "wb") as f:
        while True:
            r = _req("GET", "/read", params={"path": remote, "offset": off}, timeout=180)
            data = base64.b64decode(r["b64"])
            f.write(data)
            off += len(data)
            if r["eof"]:
                return off


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("command")
    rp.add_argument("--cwd")
    rp.add_argument("--timeout", type=int, default=7200)
    bp = sub.add_parser("bg")
    bp.add_argument("command")
    bp.add_argument("--cwd")
    pp = sub.add_parser("poll")
    pp.add_argument("job")
    pp.add_argument("--offset", type=int, default=0)
    up = sub.add_parser("upload")
    up.add_argument("local")
    up.add_argument("remote")
    dn = sub.add_parser("download")
    dn.add_argument("remote")
    dn.add_argument("local")
    sub.add_parser("health")
    a = p.parse_args()
    if a.cmd == "run":
        rc, _ = run(a.command, cwd=a.cwd, timeout=a.timeout)
        sys.exit(rc or 0)
    elif a.cmd == "bg":
        print(exec_bg(a.command, a.cwd))
    elif a.cmd == "poll":
        print(json.dumps(poll(a.job, a.offset), indent=2))
    elif a.cmd == "upload":
        print(upload(a.local, a.remote), "bytes")
    elif a.cmd == "download":
        print(download(a.remote, a.local), "bytes")
    elif a.cmd == "health":
        print(json.dumps(health(), indent=2))
