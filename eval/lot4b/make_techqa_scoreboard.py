#!/usr/bin/env python3
"""Scoreboard « match retour » de la mémoire-poids : lookup FACTUEL (docs techniques
post-2024) vs COMPRÉHENSION (QuALITY). Montre si la mémoire brille quand les questions
collent à son design (extraction de faits courts).

Usage: python make_techqa_scoreboard.py --results techqa_results.jsonl \
    --time "13h00 (Paris)" --out dashboard.html [--next-ms N] [--final]
"""
import argparse
import json
import os

# Référence QuALITY (volet compréhension, déjà mesuré) — sous-ensemble DUR
QUAL_HARD = {"C1": (0, 11), "C3": (3, 11), "C4": (9, 11)}

SCRIPT = ("<script>(function(){var N=__NEXT__;function p(n){return(n<10?'0':'')+n;}"
          "function t(){var e=document.getElementById('nx');if(!e)return;var d=Math.max(0,N-Date.now());"
          "var m=Math.floor(d/60000),s=Math.floor((d%60000)/1000);e.textContent=d>0?(p(m)+':'+p(s)):'maj\\u2026';}"
          "if(N){t();setInterval(t,1000);}})();</script>")


def pooled(rows):
    agg = {}
    for k in ("C0", "C1", "C3", "C4"):
        h = sum(r["configs"].get(k, {}).get("hits") or 0 for r in rows
                if isinstance(r["configs"].get(k, {}).get("hits"), int))
        n = sum(r["configs"].get(k, {}).get("n") or 0 for r in rows
                if isinstance(r["configs"].get(k, {}).get("hits"), int))
        agg[k] = {"h": h, "n": n, "acc": (h / n) if n else None}
    # sous-ensemble dur : questions que C0 rate
    hard = {k: [0, 0] for k in ("C1", "C3", "C4")}
    for r in rows:
        c0 = {p["qid"]: p["hit"] for p in r["configs"].get("C0", {}).get("per_q", [])}
        hq = {q for q, hh in c0.items() if hh == 0}
        for k in hard:
            for p in r["configs"].get(k, {}).get("per_q", []):
                if p["qid"] in hq:
                    hard[k][1] += 1
                    hard[k][0] += p["hit"]
    acq = [r["sleep"].get("acquired") for r in rows
           if isinstance(r.get("sleep", {}).get("acquired"), (int, float))]
    return agg, hard, (round(sum(acq) / len(acq), 2) if acq else None), len(rows)


def bar(acc, kind=""):
    w = round((acc or 0) * 100, 1)
    v = f"{round(acc*100)}%" if acc is not None else "—"
    return (f'<div class="bar {kind}"><i style="width:{w}%"></i></div><div class="mv">{v}</div>')


def build(rows, time_str, next_ms=0, final=False):
    agg, hard, acq, ndoc = pooled(rows) if rows else ({}, {}, None, 0)
    have = bool(rows)
    live = ('<span class="live done">terminé ✓</span>' if final else
            ('<span class="live">en direct</span>' if have else
             '<span class="live">test en cours…</span>'))
    def A(k):
        return agg.get(k, {}).get("acc") if have else None
    # panneau technique — sous-ensemble dur (le test qui compte)
    def H(k):
        h = hard.get(k)
        return f'{h[0]}/{h[1]}' if (h and h[1]) else '—'
    hn = hard.get("C1", [0, 0])[1] if have else 0
    tech_rows = ""
    for k, name, kind in [("C0", "Modèle nu", ""), ("C1", "Mémoire-poids", "claim"),
                          ("C3", "RAG", ""), ("C4", "Plein contexte", "ceil")]:
        tech_rows += (f'<div class="r"><div class="rl">{name}</div>{bar(A(k), kind)}</div>')
    # verdict
    c0, c1 = A("C0"), A("C1")
    if have and c0 is not None and c1 is not None:
        d = round((c1 - c0) * 100)
        verdict = (f'<b class="{"g" if d>0 else ("r" if d<0 else "m")}">{"+" if d>0 else ""}{d} pts</b> '
                   f'mémoire vs modèle nu' + (f' · sur les questions dures : mémoire <b>{H("C1")}</b> '
                   f'(QuALITY était 0/11)' if hn else ''))
    else:
        verdict = 'en attente des premiers résultats…'
    timer = ('' if final or not next_ms else
             '<div class="tm"><div class="tv" id="nx">--:--</div><div class="tl">prochain doc</div></div>')
    ql = QUAL_HARD
    html = TPL.format(
        live=live, tech_rows=tech_rows, verdict=verdict, timer=timer, time=time_str,
        ndoc=ndoc, acq=(f'{acq:.0%}' if acq else '…'),
        q1=f'{ql["C1"][0]}/{ql["C1"][1]}', q3=f'{ql["C3"][0]}/{ql["C3"][1]}',
        q4=f'{ql["C4"][0]}/{ql["C4"][1]}',
        thard=(f'{hn} questions' if hn else '…'))
    if next_ms and not final:
        html = html.replace("</main>", SCRIPT.replace("__NEXT__", str(int(next_ms))) + "</main>")
    return html


TPL = """<title>Mémoire-poids — le match retour</title>
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
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:18px 18px 20px;margin:0 0 14px;}}
  .card.big{{border-color:var(--accent);border-width:2px;}}
  .ct{{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:650;margin-bottom:2px;}}
  .ch{{font-size:1.02rem;font-weight:700;margin-bottom:14px;}}
  .r{{display:grid;grid-template-columns:1.1fr 3fr auto;gap:12px;align-items:center;margin:9px 0;}}
  .rl{{font-size:.86rem;}}
  .bar{{height:11px;border-radius:6px;background:var(--panel2);overflow:hidden;}}
  .bar>i{{display:block;height:100%;background:var(--accent-ink);border-radius:6px;transition:width .5s;}}
  .bar.claim>i{{background:var(--accent);}} .bar.ceil>i{{background:var(--ceil);}}
  .mv{{font-size:1.05rem;font-weight:750;font-family:ui-monospace,Consolas,monospace;text-align:right;min-width:3ch;}}
  .verdict{{background:var(--panel2);border-radius:12px;padding:13px 16px;text-align:center;font-size:.9rem;margin-top:6px;}}
  .verdict b{{font-family:ui-monospace,Consolas,monospace;}}
  .g{{color:var(--good);}} .r2{{color:var(--bad);}} .m{{color:var(--muted);}}
  .qhard{{display:flex;gap:10px;margin-top:8px;}}
  .qh{{flex:1;text-align:center;background:var(--panel2);border-radius:10px;padding:10px 6px;}}
  .qh b{{font-size:1.3rem;font-weight:750;font-family:ui-monospace,Consolas,monospace;display:block;color:var(--muted);}}
  .qh.claim b{{color:var(--bad);}} .qh.ceil b{{color:var(--good);}}
  .qh span{{font-size:.62rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);}}
  .tm{{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:12px;text-align:center;margin:14px auto 0;max-width:180px;}}
  .tv{{font-size:1.4rem;font-weight:700;font-family:ui-monospace,Consolas,monospace;color:var(--accent-ink);}}
  .tl{{font-size:.64rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-top:5px;}}
  .stamp{{text-align:center;margin-top:14px;font-size:.73rem;color:var(--muted);font-family:ui-monospace,Consolas,monospace;}}
  footer{{margin-top:18px;font-size:.75rem;color:var(--muted);line-height:1.6;border-top:1px solid var(--line);padding-top:14px;}}
  footer b{{color:var(--ink);}}
</style>
<main>
  <header>
    <div class="eyebrow">Claim A · mémoire-poids · {live}</div>
    <h1>Mémoire-poids : <span class="hl">le match retour</span></h1>
  </header>
  <div class="sub">Est-ce que la mémoire brille quand les questions collent à son design (faits courts) ?</div>

  <div class="card big">
    <div class="ct">Le test — docs techniques post-2024 · lookup factuel · {ndoc} doc(s) · gate {acq}</div>
    <div class="ch">Rappel closed-book (dates 2026, versions, codes que le modèle nu ne connaît pas)</div>
    {tech_rows}
    <div class="verdict">{verdict}</div>
  </div>

  <div class="card">
    <div class="ct">Rappel — le match aller (QuALITY, compréhension) · déjà mesuré</div>
    <div class="ch">Sur les 11 questions dures que le modèle nu rate — qui les récupère ?</div>
    <div class="qhard">
      <div class="qh claim"><b>{q1}</b><span>mémoire</span></div>
      <div class="qh"><b>{q3}</b><span>RAG</span></div>
      <div class="qh ceil"><b>{q4}</b><span>plein contexte</span></div>
    </div>
  </div>
  {timer}
  <div class="stamp">MàJ {time}</div>
  <footer>
    Même mémoire (<b>/sleep</b> → LoRA), deux régimes de questions. En haut :
    <b>lookup factuel</b> (le design de la mémoire) sur des changelogs 2026 ; le
    contrôle « C0-doit-échouer » est naturel (post-2024). En bas : la
    <b>compréhension</b> QuALITY où la mémoire récupérait <b>0/11</b> des questions
    dures. On compare, honnêtement, si le type de question était le vrai problème.
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
    print(f"-> {a.out} ({len(rows)} docs)")
