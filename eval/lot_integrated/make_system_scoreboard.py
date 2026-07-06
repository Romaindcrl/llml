#!/usr/bin/env python3
"""Scoreboard du SYSTÈME LLML complet (test intégré) depuis system_results.json.

Le vrai « LLML au complet » : routeur classify() + mémoire replay + RAG + base,
sur un flux MIXTE (rappel factuel + génération de code). 4 configs :
  C0     base nue                       (aucun LLML)
  C1     LLML complet, le routeur décide (LE système)
  C2     mémoire toujours active         (routeur off — mode d'échec)
  oracle routage parfait                 (borne haute du routeur)

Deux axes mesurés par config : RAPPEL closed-book (%) et GÉNÉRATION pass@1 (%).
Le routeur « gagne sa place » si C1 ≈ oracle, ≥ C0 en rappel, et NE dégrade PAS
la génération là où C2 (toujours-mémoire) la casse.

Usage : python make_system_scoreboard.py --results system_results.json \
    --time "03h30 (Paris)" --out dashboard.html [--next-ms N] [--final]
"""
import argparse
import json
import os

CFG = [
    ("C0", "Base nue", "aucun LLML", ""),
    ("C1", "LLML complet", "le routeur décide par requête", "claim"),
    ("C2", "Mémoire forcée", "routeur désactivé (mode d'échec)", "bad"),
    ("oracle", "Routage oracle", "routage parfait (borne haute)", "ceil"),
]

SCRIPT = """
<script>(function(){var N=__NEXT__;function p(n){return(n<10?'0':'')+n;}
function t(){var e=document.getElementById('nx');if(!e)return;var d=Math.max(0,N-Date.now());
var m=Math.floor(d/60000),s=Math.floor((d%60000)/1000);e.textContent=d>0?(p(m)+':'+p(s)):'maj\\u2026';}
if(N){t();setInterval(t,1000);}})();</script>"""


def build(d, time_str, next_ms=0, final=False):
    cfgs = (d or {}).get("configs", {})
    rt = (d or {}).get("routing", {})
    sl = (d or {}).get("sleep", {})
    ndocs = (d or {}).get("n_docs", 0)
    nfacts = (d or {}).get("n_facts", 0)

    def cell(k):
        c = cfgs.get(k)
        if not c:
            return ("—", "—", 0, 0)
        r = c.get("recall_acc")
        g = c.get("gen_pass_at_1")
        return (f"{round(r*100)}%" if r is not None else "—",
                f"{round(g*100)}%" if g is not None else "—",
                (r or 0) * 100, (g or 0) * 100)

    cards = ""
    for k, name, sub, cls in CFG:
        rv, gv, rw, gw = cell(k)
        cards += (
            f'<div class="cd {cls}"><div class="cn">{name}</div>'
            f'<div class="cs">{sub}</div>'
            f'<div class="mt"><div class="ml">rappel</div>'
            f'<div class="bar"><i style="width:{rw}%"></i></div><div class="mv">{rv}</div></div>'
            f'<div class="mt"><div class="ml">génération</div>'
            f'<div class="bar g"><i style="width:{gw}%"></i></div><div class="mv">{gv}</div></div>'
            f'</div>\n')

    # routage
    route = "—"
    if rt.get("recall_total"):
        rok = rt.get("recall_correct", 0) + rt.get("gen_correct", 0)
        rtot = rt.get("recall_total", 0) + rt.get("gen_total", 0)
        route = f"{rok}/{rtot}"

    # verdicts (remplis si données présentes)
    verdicts = ""
    c0, c1, c2, orc = cfgs.get("C0"), cfgs.get("C1"), cfgs.get("C2"), cfgs.get("oracle")
    if c0 and c1:
        dr = round((c1["recall_acc"] - c0["recall_acc"]) * 100)
        vv = "g" if dr > 0 else ("r" if dr < 0 else "m")
        verdicts += (f'<div class="vd"><b class="{vv}">{"+" if dr>0 else ""}{dr} pt</b>'
                     f'<span>rappel — LLML complet vs base</span></div>')
    if c2 and c0:
        dg = round((c2["gen_pass_at_1"] - c0["gen_pass_at_1"]) * 100)
        vv = "r" if dg < 0 else "m"
        verdicts += (f'<div class="vd"><b class="{vv}">{"+" if dg>0 else ""}{dg} pt</b>'
                     f'<span>génération — mémoire forcée vs base (le risque)</span></div>')
    if c1 and orc:
        gap = round(abs(c1["recall_acc"] - orc["recall_acc"]) * 100)
        verdicts += (f'<div class="vd"><b class="m">{gap} pt</b>'
                     f'<span>écart routeur vs oracle (rappel)</span></div>')

    live = ('<span class="live done">terminé ✓</span>' if final
            else '<span class="live">assemblage en cours…</span>')
    timer = ("" if final or not next_ms else
             '<div class="tm"><div class="tv" id="nx">--:--</div>'
             '<div class="tl">prochaine mesure</div></div>')
    state = (f"{ndocs} docs internalisés · {nfacts} faits en mémoire replay · "
             f"gate {'acquise' if sl.get('committed') else '…'}" if nfacts else
             "le système s'assemble : ingestion des docs → /sleep (mémoire replay)…")
    note = ("<b>Résultat définitif.</b> Le routeur gagne sa place s'il fait le rappel "
            "SANS casser la génération (là où la mémoire forcée, C2, la dégrade)."
            if final else
            "Les colonnes se remplissent config par config (base → LLML → mémoire forcée "
            "→ oracle) à mesure que le système est évalué. Rappel closed-book (choix "
            "multiple) + génération de code (EvalPlus officiel).")

    html = TPL.format(live=live, cards=cards, route=route, verdicts=verdicts,
                      timer=timer, state=state, time=time_str, note=note)
    if next_ms and not final:
        html = html.replace("</main>", SCRIPT.replace("__NEXT__", str(int(next_ms))) + "</main>")
    return html


TPL = """<title>LLML — le système complet</title>
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
  main{{width:100%;max-width:820px;padding:28px 20px 40px;}}
  header{{text-align:center;margin-bottom:6px;}}
  h1{{margin:0;font-size:1.35rem;font-weight:700;text-wrap:balance;}}
  h1 .hl{{color:var(--accent);}}
  .eyebrow{{font-size:.72rem;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);font-weight:650;margin-bottom:8px;}}
  .live{{display:inline-flex;align-items:center;gap:7px;color:var(--accent-ink);font-weight:650;}}
  .live::before{{content:"";width:8px;height:8px;border-radius:50%;background:var(--accent);animation:pulse 1.5s ease-in-out infinite;}}
  .live.done::before{{animation:none;}}
  @keyframes pulse{{50%{{opacity:.25;}}}}
  .sub{{text-align:center;color:var(--muted);font-size:.86rem;margin:8px 0 4px;}}
  .state{{text-align:center;color:var(--muted);font-size:.78rem;margin:0 0 18px;font-family:ui-monospace,Consolas,monospace;}}
  .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}}
  .cd{{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:14px 12px;}}
  .cd.claim{{border-color:var(--accent);border-width:2px;}}
  .cd.bad{{border-color:var(--bad);}}
  .cn{{font-size:.92rem;font-weight:700;}}
  .cs{{font-size:.68rem;color:var(--muted);min-height:2.4em;margin-bottom:8px;}}
  .mt{{margin-top:8px;}}
  .ml{{font-size:.64rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);}}
  .bar{{height:8px;border-radius:4px;background:var(--panel2);overflow:hidden;margin:3px 0;}}
  .bar>i{{display:block;height:100%;background:var(--accent-ink);border-radius:4px;transition:width .4s;}}
  .bar.g>i{{background:var(--ceil);}}
  .mv{{font-size:1.15rem;font-weight:750;font-family:ui-monospace,Consolas,monospace;text-align:right;}}
  .rt{{display:flex;justify-content:center;gap:8px;align-items:baseline;margin:18px 0 6px;font-size:.9rem;color:var(--muted);}}
  .rt b{{font-size:1.3rem;color:var(--ink);font-family:ui-monospace,Consolas,monospace;}}
  .vds{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 0;}}
  .vd{{flex:1;min-width:150px;background:var(--panel2);border-radius:12px;padding:12px 14px;text-align:center;}}
  .vd b{{font-size:1.4rem;font-weight:750;font-family:ui-monospace,Consolas,monospace;display:block;}}
  .vd span{{font-size:.68rem;color:var(--muted);}}
  .g{{color:var(--good);}} .r{{color:var(--bad);}} .m{{color:var(--muted);}}
  .tm{{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:12px;text-align:center;margin:16px auto 0;max-width:200px;}}
  .tv{{font-size:1.5rem;font-weight:700;font-family:ui-monospace,Consolas,monospace;color:var(--accent-ink);}}
  .tl{{font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-top:5px;}}
  footer{{margin-top:20px;font-size:.76rem;color:var(--muted);line-height:1.6;border-top:1px solid var(--line);padding-top:14px;}}
  footer b{{color:var(--ink);}}
  @media (max-width:620px){{.grid{{grid-template-columns:repeat(2,1fr);}}}}
</style>
<main>
  <header>
    <div class="eyebrow">Système intégré · routeur + mémoire + RAG + base · {live}</div>
    <h1>LLML <span class="hl">au complet</span> — un flux mixte, le système décide</h1>
  </header>
  <div class="sub">Flux MIXTE : rappel factuel (docs internalisés) + génération de code · le routeur envoie chaque requête au bon endroit</div>
  <div class="state">{state}</div>
  <div class="grid">{cards}</div>
  <div class="rt"><span>Précision de routage :</span> <b class="num">{route}</b> <span>requêtes bien aiguillées</span></div>
  <div class="vds">{verdicts}</div>
  {timer}
  <div class="stamp" style="text-align:center;margin-top:14px;font-size:.74rem;color:var(--muted);font-family:ui-monospace,Consolas,monospace">MàJ {time}</div>
  <footer>
    Test du <b>système LLML entier</b> : le routeur (<code>classify()</code>) décide par requête
    — rappel → mémoire-poids, génération → base+RAG. On compare <b>C0</b> (base nue),
    <b>C1</b> (LLML complet, le système), <b>C2</b> (mémoire toujours active, routeur off)
    et <b>oracle</b> (routage parfait). Rappel = choix multiple déterministe ; génération =
    pass@1 EvalPlus officiel. {note}
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
    d = {}
    if os.path.exists(a.results):
        try:
            d = json.load(open(a.results, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            d = {}
    open(a.out, "w", encoding="utf-8").write(build(d, a.time, a.next_ms, a.final))
    print(f"-> {a.out} (configs: {list(d.get('configs', {}))})")
