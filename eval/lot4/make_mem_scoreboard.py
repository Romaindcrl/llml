#!/usr/bin/env python3
"""Scoreboard live Claim A (mémoire-poids) depuis memory_results.jsonl.

Comparaison closed-book à 4 configs sur les mêmes questions QuALITY :
  C0 modèle nu · C1 mémoire-poids (/sleep LoRA) · C3 RAG · C4 plein contexte.

Usage : python eval/lot4/make_mem_scoreboard.py --results memory_results.jsonl \
    --time "23h59 (Paris)" --out dashboard.html [--next-ms N] [--preliminary]
"""
import argparse
import json


CFG = [
    ("C0", "Modèle nu", "sans le document (closed-book)", "base"),
    ("C1", "Mémoire-poids", "/sleep → LoRA, sans le document", "claim"),
    ("C3", "RAG", "passages récupérés en contexte", "rag"),
    ("C4", "Plein contexte", "document entier dans le prompt", "ceil"),
]

SCRIPT = """
<script>
(function(){
  var NEXT=__NEXT__;
  function pad(n){return (n<10?'0':'')+n;}
  function tick(){var el=document.getElementById('nextt');if(!el)return;
    var d=Math.max(0,NEXT-Date.now());var m=Math.floor(d/60000),s=Math.floor((d%60000)/1000);
    el.textContent=d>0?(pad(m)+':'+pad(s)):'maj\\u2026';}
  if(NEXT){tick();setInterval(tick,1000);}
})();
</script>"""


def agg(results):
    docs = [r for r in results if r.get("configs")]
    pooled = {}
    for k, _, _, _ in CFG:
        hits = sum(r["configs"].get(k, {}).get("hits") or 0 for r in docs
                   if isinstance(r["configs"].get(k, {}).get("hits"), int))
        n = sum(r["configs"].get(k, {}).get("n") or 0 for r in docs
                if isinstance(r["configs"].get(k, {}).get("hits"), int))
        pooled[k] = {"hits": hits, "n": n, "acc": (hits / n) if n else None}
    committed = sum(1 for r in docs if r.get("sleep", {}).get("committed"))
    acq = [r["sleep"]["acquired"] for r in docs
           if isinstance(r.get("sleep", {}).get("acquired"), (int, float))]
    # coût médian (chars de prompt & latence) C1 vs C4
    def med(k, field):
        vals = [r["configs"].get(k, {}).get(field) for r in docs
                if isinstance(r["configs"].get(k, {}).get(field), (int, float))]
        return sorted(vals)[len(vals) // 2] if vals else None
    # sous-ensemble DUR : questions que le modèle nu (C0) rate — le vrai test de
    # la mémoire (C0 QuALITY est haut, beaucoup de MC devinables).
    hard = {k: [0, 0] for k in ("C0", "C1", "C3", "C4")}
    for r in docs:
        c0pq = {p["qid"]: p["hit"] for p in r["configs"].get("C0", {}).get("per_q", [])}
        hardq = {qid for qid, h in c0pq.items() if h == 0}
        for k in hard:
            for p in r["configs"].get(k, {}).get("per_q", []):
                if p["qid"] in hardq:
                    hard[k][1] += 1
                    hard[k][0] += p["hit"]
    return {
        "n_docs": len(docs), "pooled": pooled, "committed": committed,
        "acq_mean": round(sum(acq) / len(acq), 3) if acq else None,
        "hard": {k: {"hits": v[0], "n": v[1]} for k, v in hard.items()},
        "cost": {"c1_chars": med("C1", "prompt_chars_med"),
                 "c4_chars": med("C4", "prompt_chars_med"),
                 "c1_lat": med("C1", "lat_ms_med"), "c4_lat": med("C4", "lat_ms_med")},
    }


def build(results, time_str, next_ms=0, preliminary=True):
    a = agg(results)
    p = a["pooled"]
    nq = p["C0"]["n"]
    def pct(k):
        v = p[k]["acc"]
        return round(v * 100, 1) if v is not None else None
    c0, c1, c3, c4 = pct("C0"), pct("C1"), pct("C3"), pct("C4")
    ratio = round(c1 / c4, 2) if (c1 and c4) else None
    bars = ""
    for k, name, sub, kind in CFG:
        v = pct(k)
        w = v if v is not None else 0
        val = f"{v:.0f}%" if v is not None else "—"
        cls = " claim" if kind == "claim" else (" ceil" if kind == "ceil" else "")
        bars += (f'<div class="row{cls}"><div class="rl"><b>{name}</b>'
                 f'<span>{sub}</span></div>'
                 f'<div class="bar"><i style="width:{w}%"></i></div>'
                 f'<div class="rv num">{val}</div></div>\n')
    # verdicts
    v_mem = ("gagne" if (c1 and c0 and c1 > c0 + 1) else
             ("égalité" if (c1 is not None and c0 is not None and abs(c1 - c0) <= 1) else "en retrait"))
    live = ('<span class="live">préliminaire</span>' if preliminary
            else '<span class="live done">terminé ✓</span>')
    timer = ('<div class="tmr"><div class="tval" id="nextt">--:--</div>'
             '<div class="tlab">prochain relevé</div></div>') if next_ms else ""
    cst = a["cost"]
    cost_html = ""
    if cst["c1_chars"] and cst["c4_chars"]:
        red = round(cst["c4_chars"] / max(cst["c1_chars"], 1), 1)
        cost_html = (f'<div class="cost"><b>Coût du prompt</b> — mémoire-poids '
                     f'<b class="num">{cst["c1_chars"]}</b> car. vs plein contexte '
                     f'<b class="num">{cst["c4_chars"]}</b> car. '
                     f'(<b class="g">÷{red}</b> de contexte à lire).</div>')
    gate = (f'{a["committed"]}/{a["n_docs"]} docs acquis'
            + (f' (rappel des faits {a["acq_mean"]:.0%})' if a["acq_mean"] else ''))
    # sous-ensemble dur : le test décisif (questions que le modèle nu rate)
    hard = a.get("hard", {})
    hn = hard.get("C0", {}).get("n", 0)
    hard_html = ""
    if hn:
        def hcell(k, lab, cls):
            hh = hard.get(k, {}).get("hits", 0)
            return (f'<div class="hc {cls}"><b class="num">{hh}<span>/{hn}</span></b>'
                    f'<span class="hl">{lab}</span></div>')
        hard_html = (
            '<div class="hard"><div class="hh">Le test décisif — questions que le '
            f'modèle nu <b>rate</b> ({hn}) : qui les récupère&nbsp;?</div>'
            '<div class="hrow">'
            + hcell("C1", "mémoire-poids", "claim")
            + hcell("C3", "RAG", "")
            + hcell("C4", "plein contexte", "ceil")
            + '</div></div>')
    html = TPL.format(
        live=live, nq=nq, ndocs=a["n_docs"], bars=bars, ratio=(ratio if ratio else "—"),
        c1=c1 if c1 is not None else "—", c0=c0 if c0 is not None else "—",
        c4=c4 if c4 is not None else "—", vmem=v_mem, gate=gate, time=time_str,
        timer=timer, cost=cost_html, hard=hard_html,
        note=("Chiffres <b>préliminaires</b> tant que le run des docs n'est pas complet."
              if preliminary else
              "<b>Résultat définitif</b> — verdict §4.2 : « C1 &gt; C0 » et « C1 ≥ 80% de C4 » "
              "réfutées ; la mémoire-poids ne récupère aucune question dure. Elle acquiert "
              "ses faits (gate 84%) à ÷39 de contexte, mais ne restitue pas le document."),
        vcls=("g" if v_mem == "gagne" else ("m" if v_mem == "égalité" else "r")))
    if next_ms:
        html = html.replace("</main>", SCRIPT.replace("__NEXT__", str(int(next_ms))) + "</main>")
    return html


TPL = """<title>LLML — mémoire-poids (Claim A)</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{{--bg:#F6F8F7;--panel:#FFF;--panel2:#EFF3F1;--ink:#141A18;--muted:#5D6864;--line:#DCE3E0;
    --accent:#0E7A6E;--accent-ink:#0A5A51;--good:#1B8A50;--bad:#C0453E;--ceil:#9A7B22;
    --shadow:0 1px 3px rgba(15,30,25,.07);}}
  @media (prefers-color-scheme:dark){{:root{{--bg:#0E1311;--panel:#161C19;--panel2:#1D2420;--ink:#E9EFEC;
    --muted:#93A09B;--line:#263029;--accent:#3AC7B2;--accent-ink:#7FDDCE;--good:#54CD86;--bad:#E77A72;
    --ceil:#D9B84A;--shadow:0 1px 3px rgba(0,0,0,.4);}}}}
  :root[data-theme="dark"]{{--bg:#0E1311;--panel:#161C19;--panel2:#1D2420;--ink:#E9EFEC;--muted:#93A09B;
    --line:#263029;--accent:#3AC7B2;--accent-ink:#7FDDCE;--good:#54CD86;--bad:#E77A72;--ceil:#D9B84A;--shadow:0 1px 3px rgba(0,0,0,.4);}}
  :root[data-theme="light"]{{--bg:#F6F8F7;--panel:#FFF;--panel2:#EFF3F1;--ink:#141A18;--muted:#5D6864;
    --line:#DCE3E0;--accent:#0E7A6E;--accent-ink:#0A5A51;--good:#1B8A50;--bad:#C0453E;--ceil:#9A7B22;--shadow:0 1px 3px rgba(15,30,25,.07);}}
  *{{box-sizing:border-box;}}
  body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    display:flex;flex-direction:column;align-items:center;min-height:100vh;}}
  .num{{font-variant-numeric:tabular-nums;}}
  main{{width:100%;max-width:720px;padding:28px 20px 40px;}}
  header{{text-align:center;margin-bottom:18px;}}
  h1{{margin:0;font-size:1.3rem;font-weight:700;text-wrap:balance;}}
  .eyebrow{{font-size:.72rem;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);font-weight:650;margin-bottom:8px;}}
  .live{{display:inline-flex;align-items:center;gap:7px;color:var(--accent-ink);font-weight:650;}}
  .live::before{{content:"";width:8px;height:8px;border-radius:50%;background:var(--accent);animation:pulse 1.5s ease-in-out infinite;}}
  .live.done::before{{animation:none;}}
  @keyframes pulse{{50%{{opacity:.25;}}}}
  .sub{{text-align:center;color:var(--muted);font-size:.86rem;margin:2px 0 18px;}}
  .rows{{display:flex;flex-direction:column;gap:12px;margin:18px 0;}}
  .row{{display:grid;grid-template-columns:1fr 2.2fr auto;gap:12px;align-items:center;
    background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:13px 16px;}}
  .row.claim{{border-color:var(--accent);border-width:2px;}}
  .rl b{{font-size:.95rem;}} .rl span{{display:block;font-size:.72rem;color:var(--muted);}}
  .bar{{height:12px;border-radius:6px;background:var(--panel2);overflow:hidden;}}
  .bar>i{{display:block;height:100%;background:var(--accent);border-radius:6px;transition:width .4s;}}
  .row.claim .bar>i{{background:var(--accent-ink);}}
  .row.ceil .bar>i{{background:var(--ceil);}}
  .rv{{font-size:1.3rem;font-weight:750;font-family:ui-monospace,Consolas,monospace;min-width:3ch;text-align:right;}}
  .kpis{{display:flex;gap:12px;margin:16px 0;flex-wrap:wrap;}}
  .kpi{{flex:1;min-width:140px;background:var(--panel2);border-radius:12px;padding:14px 16px;text-align:center;}}
  .kpi b{{font-size:1.5rem;font-weight:750;font-family:ui-monospace,Consolas,monospace;display:block;}}
  .kpi span{{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;}}
  .g{{color:var(--good);}} .r{{color:var(--bad);}} .m{{color:var(--muted);}}
  .tmr{{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:12px;text-align:center;min-width:120px;}}
  .tval{{font-size:1.5rem;font-weight:700;font-family:ui-monospace,Consolas,monospace;color:var(--accent-ink);line-height:1;}}
  .tlab{{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-top:6px;}}
  .cost{{background:var(--panel2);border-radius:12px;padding:12px 16px;font-size:.84rem;margin:6px 0 0;}}
  .hard{{margin:16px 0 0;border:1px solid var(--bad);border-radius:12px;padding:14px 16px;background:var(--panel);box-shadow:var(--shadow);}}
  .hard .hh{{font-size:.85rem;color:var(--ink);margin-bottom:12px;text-align:center;}}
  .hard .hh b{{color:var(--bad);}}
  .hrow{{display:flex;gap:10px;}}
  .hc{{flex:1;text-align:center;background:var(--panel2);border-radius:10px;padding:12px 8px;}}
  .hc b{{font-size:1.7rem;font-weight:750;font-family:ui-monospace,Consolas,monospace;display:block;line-height:1;color:var(--muted);}}
  .hc b span{{font-size:.9rem;color:var(--muted);}}
  .hc.claim b{{color:var(--bad);}} .hc.ceil b{{color:var(--good);}}
  .hc .hl{{font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-top:6px;display:block;}}
  .stamp{{text-align:center;margin-top:14px;font-size:.74rem;color:var(--muted);font-family:ui-monospace,Consolas,monospace;}}
  footer{{margin-top:20px;font-size:.76rem;color:var(--muted);line-height:1.6;border-top:1px solid var(--line);padding-top:14px;}}
  footer b{{color:var(--ink);}}
  @media (max-width:460px){{.row{{grid-template-columns:1fr auto;}}.bar{{grid-column:1/-1;order:3;}}.rv{{font-size:1.1rem;}}h1{{font-size:1.1rem;}}}}
</style>
<main>
  <header>
    <div class="eyebrow">Claim A · mémoire-poids · QuALITY · {live}</div>
    <h1>Lire depuis les <span style="color:var(--accent)">poids</span> vs depuis le contexte</h1>
  </header>
  <div class="sub">Rappel closed-book sur {nq} questions · {ndocs} doc(s) · scoring déterministe (choix multiple)</div>
  <div class="rows">{bars}</div>
  <div class="kpis">
    <div class="kpi"><b class="{vcls}">{c1}%</b><span>Mémoire-poids (C1)</span></div>
    <div class="kpi"><b>{ratio}</b><span>ratio C1 / plein contexte</span></div>
    <div class="kpi"><b class="m">{c0}%</b><span>Modèle nu (C0)</span></div>
    {timer}
  </div>
  {cost}
  {hard}
  <div class="stamp">gate /sleep : {gate} · MàJ {time}</div>
  <footer>
    Test de <b>Claim A</b> (CDC §4.2) : un document internalisé dans un LoRA via
    <b>/sleep</b> est-il restituable <b>closed-book</b> (document retiré du prompt) ?
    On compare, sur les MÊMES questions QuALITY : modèle nu (C0), mémoire-poids (C1,
    le claim), RAG (C3), et plein contexte (C4, borne haute honnête). Hypothèse
    pré-enregistrée : C1 ≥ 80% de C4 et C1 &gt; C0. Scoring déterministe (greedy,
    lettre parsée). {note}
  </footer>
</main>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--next-ms", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    import os
    rows = []
    if os.path.exists(a.results):
        rows = [json.loads(l) for l in open(a.results, encoding="utf-8") if l.strip()]
    open(a.out, "w", encoding="utf-8").write(
        build(rows, a.time, a.next_ms, preliminary=not a.final))
    print(f"-> {a.out} ({len(rows)} docs)")
