#!/usr/bin/env python3
"""Scoreboard de la BOUCLE d'apprentissage autonome (Lot 6bis, bench #12) — la
stack LLML VIVANTE en continu. Montre, cycle par cycle : ce que le système apprend
(C1 intégré vs C0 base vs RAG-oracle), l'oubli, la non-régression génération, et
le verdict live des 4 hypothèses pré-enregistrées H-D1..D4.

Usage: python make_autoloop_scoreboard.py --results autoloop_results.jsonl \
    --time "15h00 (Paris)" --out dashboard.html [--next-ms N] [--final]
"""
import argparse
import json
import os

SCRIPT = ("<script>(function(){var N=__NEXT__;function p(n){return(n<10?'0':'')+n;}"
          "function t(){var e=document.getElementById('nx');if(!e)return;var d=Math.max(0,N-Date.now());"
          "var m=Math.floor(d/60000),s=Math.floor((d%60000)/1000);e.textContent=d>0?(p(m)+':'+p(s)):'maj\\u2026';}"
          "if(N){t();setInterval(t,1000);}})();</script>")


def acc(d):
    return (sum(d.values()) / len(d)) if d else None


def analyze(rows):
    """Extrait les métriques clés pour les hypothèses."""
    if not rows:
        return {}
    last = rows[-1]
    c1_qids = list(rows[0]["per_q"]["c1"].keys())          # QA du cycle 1
    # oubli : rappel C1 des QA du cycle 1, mesuré au cycle 1 puis au dernier cycle
    def cyc1_recall(row):
        c1 = row["per_q"]["c1"]
        sub = {q: c1[q] for q in c1_qids if q in c1}
        return acc(sub), sum(sub.values()), len(sub)
    r1 = cyc1_recall(rows[0])
    rL = cyc1_recall(last)
    gen0 = rows[0]["gen_nonreg"]
    genL = last["gen_nonreg"]
    return {
        "c0": last["acc"]["C0"], "c1": last["acc"]["C1"], "oracle": last["acc"]["RAG_oracle"],
        "n_seen": last["n_seen"], "n_cycles": len(rows),
        "forget_1": r1, "forget_L": rL,
        "gen0": gen0, "genL": genL,
        "committed": [bool(r["sleep"].get("committed")) for r in rows],
        "ltm": last["ltm_facts"],
    }


def hyp_verdict(m):
    """Verdicts live des 4 hypothèses."""
    if not m:
        return {k: ("wait", "en attente…") for k in ("D1", "D2", "D3", "D4")}
    c0, c1, orc = m["c0"], m["c1"], m["oracle"]
    d1 = round((c1 - c0) * 100)
    v1 = ("ok" if d1 >= 20 else "no", f"C1 {c1:.0%} vs C0 {c0:.0%} = "
          f"{'+' if d1>=0 else ''}{d1} pts")
    gL, g0 = m["genL"], m["gen0"]
    okg = gL["pass"] >= g0["pass"] - 1
    v2 = ("ok" if okg else "no", f"génération {gL['pass']}/{gL['n']} préservée "
          f"(base {g0['pass']}/{g0['n']})")
    (a1, h1, n1), (aL, hL, nL) = m["forget_1"], m["forget_L"]
    if n1:
        okf = (aL or 0) >= (a1 or 0) - (1.0 / max(1, n1))
        v3 = ("ok" if okf else "no", f"QA du cycle 1 : {hL}/{nL} après tous les /sleep "
              f"(était {h1}/{n1})")
    else:
        v3 = ("wait", "…")
    d4 = round((orc - c1) * 100)
    v4 = ("info", f"RAG-oracle {orc:.0%} vs C1 {c1:.0%} = "
          f"{'+' if d4>=0 else ''}{d4} pts" + (" → routage rappel→poids sous-optimal" if d4 >= 15 else ""))
    return {"D1": v1, "D2": v2, "D3": v3, "D4": v4}


def cycle_rows_html(rows):
    if not rows:
        return ('<div class="crow head"><span>cycle</span><span>versions apprises</span>'
                '<span>LTM</span><span>gate</span><span>C0</span><span>C1</span>'
                '<span>RAG*</span></div>'
                '<div class="empty">boucle en cours de démarrage…</div>')
    out = ('<div class="crow head"><span>cycle</span><span>versions apprises</span>'
           '<span>LTM</span><span>gate</span><span>C0</span><span>C1</span><span>RAG*</span></div>')
    for r in rows:
        sl = r["sleep"]
        gate = f'{sl.get("acquired", 0):.0%}' if sl.get("acquired") is not None else "—"
        gc = "cok" if sl.get("committed") else "cno"
        vers = ", ".join(v.replace(" 0.", " ") for v in r["versions"])[:34]
        def cell(k, cls=""):
            v = r["acc"].get(k)
            return f'<span class="{cls}">{v:.0%}</span>' if v is not None else '<span>—</span>'
        out += (f'<div class="crow"><span class="cy">{r["cycle"]}</span>'
                f'<span class="vs">{vers}</span>'
                f'<span class="num">{r["ltm_facts"]}</span>'
                f'<span class="num {gc}">{gate}</span>'
                f'{cell("C0")}{cell("C1","c1")}{cell("RAG_oracle","orc")}</div>')
    return out


HYP_META = {
    "D1": "Le système apprend (C1 &gt; C0)",
    "D2": "Pas de régression génération",
    "D3": "Pas d'oubli catastrophique",
    "D4": "Intégration vs RAG-oracle",
}
BADGE = {"ok": ("✓", "g"), "no": ("✗", "r2"), "info": ("•", "m"), "wait": ("…", "m")}


def build(rows, time_str, next_ms=0, final=False):
    m = analyze(rows)
    hv = hyp_verdict(m)
    have = bool(rows)
    live = ('<span class="live done">terminé ✓</span>' if final else
            ('<span class="live">en direct · cycle %d/5</span>' % len(rows) if have else
             '<span class="live">démarrage…</span>'))
    hyps = ""
    for k in ("D1", "D2", "D3", "D4"):
        state, txt = hv[k]
        sym, cls = BADGE[state]
        hyps += (f'<div class="hyp"><div class="hb {cls}">{sym}</div>'
                 f'<div class="ht"><b>H-{k}</b> — {HYP_META[k]}<span>{txt}</span></div></div>')
    timer = ('' if final or not next_ms else
             '<div class="tm"><div class="tv" id="nx">--:--</div><div class="tl">prochain cycle</div></div>')
    ltm = m.get("ltm", "…") if have else "…"
    ncy = m.get("n_cycles", 0)
    html = TPL.format(live=live, cycles=cycle_rows_html(rows), hyps=hyps, timer=timer,
                      time=time_str, ltm=ltm, ncy=ncy)
    if next_ms and not final:
        html = html.replace("</main>", SCRIPT.replace("__NEXT__", str(int(next_ms))) + "</main>")
    return html


TPL = """<title>LLML — la stack entière, vivante</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{{--bg:#F6F8F7;--panel:#FFF;--panel2:#EFF3F1;--ink:#141A18;--muted:#5D6864;--line:#DCE3E0;
    --accent:#0E7A6E;--accent-ink:#0A5A51;--good:#1B8A50;--bad:#C0453E;--ceil:#9A7B22;--shadow:0 1px 3px rgba(15,30,25,.07);}}
  @media (prefers-color-scheme:dark){{:root{{--bg:#0E1311;--panel:#161C19;--panel2:#1D2420;--ink:#E9EFEC;--muted:#93A09B;
    --line:#263029;--accent:#3AC7B2;--accent-ink:#7FDDCE;--good:#54CD86;--bad:#E77A72;--ceil:#D9B84A;--shadow:0 1px 3px rgba(0,0,0,.4);}}}}
  :root[data-theme="dark"]{{--bg:#0E1311;--panel:#161C19;--panel2:#1D2420;--ink:#E9EFEC;--muted:#93A09B;--line:#263029;
    --accent:#3AC7B2;--accent-ink:#7FDDCE;--good:#54CD86;--bad:#E77A72;--ceil:#D9B84A;--shadow:0 1px 3px rgba(0,0,0,.4);}}
  :root[data-theme="light"]{{--bg:#F6F8F7;--panel:#FFF;--panel2:#EFF3F1;--ink:#141A18;--muted:#5D6864;--line:#DCE3E0;
    --accent:#0E7A6E;--accent-ink:#0A5A51;--good:#1B8A50;--bad:#C0453E;--ceil:#9A7B22;--shadow:0 1px 3px rgba(15,30,25,.07);}}
  *{{box-sizing:border-box;}}
  body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    display:flex;flex-direction:column;align-items:center;min-height:100vh;}}
  .num{{font-variant-numeric:tabular-nums;}}
  main{{width:100%;max-width:720px;padding:28px 20px 40px;}}
  header{{text-align:center;margin-bottom:6px;}}
  h1{{margin:0;font-size:1.32rem;font-weight:700;text-wrap:balance;}}
  h1 .hl{{color:var(--accent);}}
  .eyebrow{{font-size:.72rem;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);font-weight:650;margin-bottom:8px;}}
  .live{{display:inline-flex;align-items:center;gap:7px;color:var(--accent-ink);font-weight:650;}}
  .live::before{{content:"";width:8px;height:8px;border-radius:50%;background:var(--accent);animation:pulse 1.5s ease-in-out infinite;}}
  .live.done::before{{animation:none;}}
  @keyframes pulse{{50%{{opacity:.25;}}}}
  .sub{{text-align:center;color:var(--muted);font-size:.85rem;margin:8px 0 18px;}}
  .flow{{display:flex;justify-content:center;flex-wrap:wrap;gap:6px;font-size:.68rem;color:var(--muted);margin:0 0 18px;}}
  .flow span{{background:var(--panel2);border-radius:20px;padding:4px 10px;font-weight:600;}}
  .flow b{{color:var(--accent-ink);}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:18px;margin:0 0 14px;}}
  .card.big{{border-color:var(--accent);border-width:2px;}}
  .ct{{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:650;margin-bottom:2px;}}
  .ch{{font-size:1.02rem;font-weight:700;margin-bottom:14px;}}
  .crow{{display:grid;grid-template-columns:2.6rem 1fr 2.2rem 2.6rem 2.4rem 2.4rem 2.4rem;gap:6px;align-items:center;
    padding:7px 0;border-top:1px solid var(--line);font-size:.82rem;}}
  .crow.head{{border-top:none;font-size:.6rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:650;}}
  .crow .cy{{font-weight:750;color:var(--accent-ink);text-align:center;}}
  .crow .vs{{font-size:.72rem;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
  .crow span:nth-child(n+3){{text-align:center;font-family:ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums;}}
  .crow .c1{{font-weight:750;color:var(--accent);}}
  .crow .orc{{font-weight:700;color:var(--ceil);}}
  .cok{{color:var(--good);}} .cno{{color:var(--bad);}}
  .empty{{text-align:center;color:var(--muted);padding:22px;font-size:.85rem;}}
  .hyp{{display:flex;gap:12px;align-items:flex-start;padding:11px 0;border-top:1px solid var(--line);}}
  .hyp:first-of-type{{border-top:none;}}
  .hb{{flex:none;width:26px;height:26px;border-radius:50%;background:var(--panel2);display:flex;align-items:center;
    justify-content:center;font-weight:800;font-size:.95rem;}}
  .hb.g{{color:var(--good);}} .hb.r2{{color:var(--bad);}} .hb.m{{color:var(--muted);}}
  .ht{{font-size:.85rem;font-weight:600;}}
  .ht span{{display:block;font-weight:500;color:var(--muted);font-size:.78rem;font-family:ui-monospace,Consolas,monospace;margin-top:2px;}}
  .g{{color:var(--good);}} .r2{{color:var(--bad);}} .m{{color:var(--muted);}}
  .tm{{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:12px;text-align:center;margin:14px auto 0;max-width:180px;}}
  .tv{{font-size:1.4rem;font-weight:700;font-family:ui-monospace,Consolas,monospace;color:var(--accent-ink);}}
  .tl{{font-size:.64rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-top:5px;}}
  .stamp{{text-align:center;margin-top:14px;font-size:.73rem;color:var(--muted);font-family:ui-monospace,Consolas,monospace;}}
  footer{{margin-top:18px;font-size:.75rem;color:var(--muted);line-height:1.6;border-top:1px solid var(--line);padding-top:14px;}}
  footer b{{color:var(--ink);}}
</style>
<main>
  <header>
    <div class="eyebrow">Lot 6bis · bench #12 · {live}</div>
    <h1>La stack LLML <span class="hl">entière, vivante</span></h1>
  </header>
  <div class="sub">Une boucle continue : l'agent travaille, son contexte sature, la mémoire se consolide toute seule, le routeur décide.</div>
  <div class="flow"><span>agent <b>travaille</b></span><span>→ contexte <b>sature</b></span>
    <span>→ <b>auto-promotion</b> LTM</span><span>→ <b>/sleep</b> (LoRA)</span><span>→ <b>routeur</b> décide</span></div>

  <div class="card big">
    <div class="ct">Progression — {ncy}/5 cycles · LTM {ltm} faits · corpus gelé uv+ruff post-2024</div>
    <div class="ch">Ce que le système sait, cycle après cycle</div>
    {cycles}
    <div style="font-size:.68rem;color:var(--muted);margin-top:10px;">
      C0 = base nue · <b style="color:var(--accent)">C1 = LLML intégré (routeur)</b> ·
      <b style="color:var(--ceil)">RAG*</b> = RAG-oracle (force base+RAG) · gate = acquisition /sleep (vert = commité)
    </div>
  </div>

  <div class="card">
    <div class="ct">Hypothèses pré-enregistrées · verdict live</div>
    <div class="ch">Ce que la boucle prouve — ou pas</div>
    {hyps}
  </div>
  {timer}
  <div class="stamp">MàJ {time}</div>
  <footer>
    Premier test de la <b>stack entière en boucle</b> : rien n'est appelé à la main.
    L'agent lit un flux de changelogs 2026, son contexte <b>sature</b>, le contenu
    libéré est <b>auto-promu</b> en mémoire, un <b>/sleep</b> gé entraîne un LoRA, puis
    le <b>routeur</b> décide par requête (rappel→poids / génération→base+RAG). On mesure
    si le tout <b>apprend</b> vraiment, <b>oublie</b>, régresse, et si router les faits
    vers les <b>poids</b> plutôt que le <b>RAG</b> est le bon choix.
  </footer>
</main>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--next-ms", type=int, default=0)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    if os.path.exists(a.results):
        rows = [json.loads(l) for l in open(a.results, encoding="utf-8") if l.strip()]
    open(a.out, "w", encoding="utf-8").write(build(rows, a.time, a.next_ms, a.final))
    print(f"-> {a.out} ({len(rows)} cycles)")
