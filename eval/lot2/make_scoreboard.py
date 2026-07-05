#!/usr/bin/env python3
"""Régénère le scoreboard live (HTML artefact) depuis un tally live_score.

Usage : python eval/lot2/make_scoreboard.py --tally tally_humaneval.json \
    --time "18:34 (Paris)" --baseline 78.0 --out dashboard.html
"""
import argparse
import json

TPL_HEAD = open(__file__.replace("make_scoreboard.py", "_scoreboard_shell.html"),
                encoding="utf-8").read() if False else None


def build(t, time_str, baseline):
    n = t.get("n_scored", 0)
    total = t.get("n_total", 164)
    ds = t.get("baseline", {})
    vs = t.get("llml", {})
    bf, lf = ds.get("plus_fail", 0), vs.get("plus_fail", 0)
    bp, lp = ds.get("plus_pass", 0), vs.get("plus_pass", 0)
    wins = len(t.get("verify_wins", []))
    regs = len(t.get("verify_regressions", []))
    delta = lp - bp
    pct = round(n / total * 100, 1) if total else 0
    ds_name = "HumanEval+" if t.get("dataset") == "humaneval" else "MBPP+"
    # qui mène
    llml_tag = '<span class="tag">✓ mène</span>' if delta > 0 else (
        '<span class="tag" style="background:var(--bad-bg);color:var(--bad)">− en retrait</span>'
        if delta < 0 else '<span class="tag" style="background:var(--panel2);color:var(--muted)">égalité</span>')
    llml_win = " win" if delta > 0 else ""
    base_win = " win" if delta < 0 else ""
    dcls = "g" if delta > 0 else ("r" if delta < 0 else "m")
    dtxt = f"+{delta}" if delta > 0 else str(delta)
    return HTML.format(
        ds=ds_name, n=n, total=total, pct=pct, time=time_str, baseline=baseline,
        bf=bf, bp=bp, lf=lf, lp=lp, wins=wins, regs=regs,
        llml_tag=llml_tag, llml_win=llml_win, base_win=base_win, dcls=dcls, dtxt=dtxt)


HTML = """<title>LLML vs Baseline — live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{{--bg:#F6F8F7;--panel:#FFFFFF;--panel2:#EFF3F1;--ink:#141A18;--muted:#5D6864;
    --line:#DCE3E0;--accent:#0E7A6E;--accent-ink:#0A5A51;--good:#1B8A50;--good-bg:#E3F3E9;
    --bad:#C0453E;--bad-bg:#F7E7E5;--shadow:0 1px 3px rgba(15,30,25,.07);}}
  @media (prefers-color-scheme:dark){{:root{{--bg:#0E1311;--panel:#161C19;--panel2:#1D2420;
    --ink:#E9EFEC;--muted:#93A09B;--line:#263029;--accent:#3AC7B2;--accent-ink:#7FDDCE;
    --good:#54CD86;--good-bg:#13301E;--bad:#E77A72;--bad-bg:#361A18;--shadow:0 1px 3px rgba(0,0,0,.4);}}}}
  :root[data-theme="dark"]{{--bg:#0E1311;--panel:#161C19;--panel2:#1D2420;--ink:#E9EFEC;
    --muted:#93A09B;--line:#263029;--accent:#3AC7B2;--accent-ink:#7FDDCE;--good:#54CD86;
    --good-bg:#13301E;--bad:#E77A72;--bad-bg:#361A18;--shadow:0 1px 3px rgba(0,0,0,.4);}}
  :root[data-theme="light"]{{--bg:#F6F8F7;--panel:#FFFFFF;--panel2:#EFF3F1;--ink:#141A18;
    --muted:#5D6864;--line:#DCE3E0;--accent:#0E7A6E;--accent-ink:#0A5A51;--good:#1B8A50;
    --good-bg:#E3F3E9;--bad:#C0453E;--bad-bg:#F7E7E5;--shadow:0 1px 3px rgba(15,30,25,.07);}}
  *{{box-sizing:border-box;}}
  body{{margin:0;background:var(--bg);color:var(--ink);
    font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    display:flex;flex-direction:column;align-items:center;min-height:100vh;}}
  .num{{font-variant-numeric:tabular-nums;}}
  main{{width:100%;max-width:760px;padding:28px 20px 40px;}}
  header{{text-align:center;margin-bottom:20px;}}
  h1{{margin:0;font-size:1.35rem;font-weight:700;text-wrap:balance;}}
  h1 .vs{{color:var(--accent);font-family:ui-monospace,Consolas,monospace;}}
  .eyebrow{{font-size:.72rem;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);
    font-weight:650;margin-bottom:8px;}}
  .live{{display:inline-flex;align-items:center;gap:7px;color:var(--accent-ink);font-weight:650;}}
  .live::before{{content:"";width:8px;height:8px;border-radius:50%;background:var(--accent);
    animation:pulse 1.5s ease-in-out infinite;}}
  @keyframes pulse{{50%{{opacity:.25;}}}}
  @media (prefers-reduced-motion:reduce){{.live::before{{animation:none;}}}}
  .prog-wrap{{margin:22px 0 26px;}}
  .prog-head{{display:flex;justify-content:space-between;align-items:baseline;
    font-size:.82rem;color:var(--muted);margin-bottom:7px;}}
  .prog-head b{{color:var(--ink);font-size:1.05rem;font-weight:700;}}
  .prog{{height:10px;border-radius:5px;background:var(--panel2);overflow:hidden;}}
  .prog>i{{display:block;height:100%;background:var(--accent);border-radius:5px;}}
  .board{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
  .col{{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    box-shadow:var(--shadow);padding:20px 18px;text-align:center;position:relative;}}
  .col.win{{border-color:var(--accent);}}
  .col h2{{margin:0 0 2px;font-size:.95rem;font-weight:700;}}
  .col .who{{font-size:.72rem;color:var(--muted);margin-bottom:16px;min-height:2.2em;}}
  .fails{{font-size:3.6rem;font-weight:750;line-height:1;color:var(--bad);
    font-family:ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums;}}
  .fails-lab{{font-size:.74rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-top:6px;}}
  .passes{{margin-top:16px;font-size:.9rem;color:var(--good);font-weight:650;}}
  .passes .num{{font-size:1.05rem;}}
  .tag{{position:absolute;top:12px;right:12px;font-size:.64rem;font-weight:700;
    text-transform:uppercase;letter-spacing:.06em;padding:3px 8px;border-radius:999px;
    background:var(--good-bg);color:var(--good);}}
  .delta{{margin-top:16px;background:var(--panel2);border-radius:12px;padding:14px 18px;
    display:flex;flex-wrap:wrap;gap:6px 22px;justify-content:center;align-items:baseline;font-size:.9rem;}}
  .delta b{{font-size:1.15rem;font-weight:750;font-family:ui-monospace,Consolas,monospace;}}
  .delta .g{{color:var(--good);}} .delta .r{{color:var(--bad);}} .delta .m{{color:var(--muted);}}
  footer{{margin-top:22px;font-size:.76rem;color:var(--muted);line-height:1.6;
    border-top:1px solid var(--line);padding-top:16px;}}
  footer b{{color:var(--ink);}}
  .stamp{{text-align:center;margin-top:14px;font-size:.74rem;color:var(--muted);
    font-family:ui-monospace,Consolas,monospace;}}
  @media (max-width:460px){{.fails{{font-size:2.9rem;}}h1{{font-size:1.12rem;}}
    main{{padding:20px 14px 32px;}}.col{{padding:16px 12px;}}}}
</style>
<main>
  <header>
    <div class="eyebrow">Claim C · boucle de vérification · {ds} · <span class="live">en direct</span></div>
    <h1>LLML <span class="vs">vs</span> modèle nu</h1>
  </header>
  <div class="prog-wrap">
    <div class="prog-head"><span>Problèmes scorés (même sous-ensemble, les deux)</span>
      <span><b class="num">{n}</b> / {total}</span></div>
    <div class="prog" role="img" aria-label="{n} sur {total} problèmes scorés"><i style="width:{pct}%"></i></div>
  </div>
  <div class="board">
    <div class="col{base_win}">
      <h2>Modèle nu</h2><div class="who">baseline C0 · draft one-shot</div>
      <div class="fails num">{bf}</div><div class="fails-lab">échecs</div>
      <div class="passes"><span class="num">{bp}</span> réussis</div>
    </div>
    <div class="col{llml_win}">
      {llml_tag}<h2>LLML</h2>
      <div class="who">C0 + verify · draft → exécute les exemples → répare</div>
      <div class="fails num">{lf}</div><div class="fails-lab">échecs</div>
      <div class="passes"><span class="num">{lp}</span> réussis</div>
    </div>
  </div>
  <div class="delta">
    <span><b class="{dcls}">{dtxt}</b> réussite(s) pour LLML</span>
    <span><b class="g">{wins}</b> <span class="m">corrigé(s)</span></span>
    <span><b class="r">{regs}</b> <span class="m">régression(s)</span></span>
  </div>
  <div class="stamp">MàJ {time} · se rafraîchit à chaque relevé — recharge la page</div>
  <footer>
    Scoring <b>officiel EvalPlus</b> (tests cachés), appliqué au sous-ensemble déjà
    généré, <b>identique pour les deux bras</b> (comparaison appariée, même timing).
    Chiffre <b>provisoire</b> : le résultat officiel sera le scoring final sur les
    {total} problèmes complets. « Échec » = ne passe pas les tests cachés {ds}.
    Baseline C0 de référence (run complet Lot 1) : <b>{baseline} %</b>.
  </footer>
</main>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tally", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--baseline", default="78,0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = json.load(open(a.tally, encoding="utf-8"))
    open(a.out, "w", encoding="utf-8").write(build(t, a.time, a.baseline))
    print(f"-> {a.out} (n_scored={t.get('n_scored')})")
