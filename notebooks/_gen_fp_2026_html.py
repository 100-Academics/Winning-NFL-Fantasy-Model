"""Render the 2026 live FantasyPros head-to-head scoreboard as a themed HTML
widget (reads models/fantasypros_2026_report.json). Self-contained, uses the
app's live CSS vars. Run: uv run python notebooks/_gen_fp_2026_html.py
Out: notebooks/fp_2026_scoreboard.html
"""
from __future__ import annotations

import html
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
rep = json.loads((ROOT / "models" / "fantasypros_2026_report.json").read_text())
pos = rep["per_position"]
weeks = rep["per_week"]
pooled = rep["pooled"]
TOPK = rep.get("top_k", 5)

POS_ORDER = ["QB", "RB", "WR", "TE"]


def badge(win: bool) -> str:
    cls = "win" if win else "lose"
    txt = "WIN" if win else "trail"
    return f'<span class="badge {cls}">{txt}</span>'


def metric_row(label: str, ours, fp, higher_is_better: bool, n: int) -> str:
    win = (ours > fp) if higher_is_better else (ours < fp)
    o = "—" if ours is None else f"{ours:.2f}" if isinstance(ours, float) else str(ours)
    f = "—" if fp is None else f"{fp:.2f}" if isinstance(fp, float) else str(fp)
    return (f'<tr><td class="lbl">{label}</td>'
            f'<td class="ours">{o}</td><td class="fp">{f}</td>'
            f'<td class="win">{badge(win)}</td><td class="n">{n}</td></tr>')


# per-position table
rows = ""
for p in POS_ORDER:
    if p not in pos:
        continue
    d = pos[p]
    raw = d["raw_stat"]
    # highlight the position header cell if it's a net winner
    wins = sum([raw["we_win_raw"], d["we_win_mae"], d["we_win_rmse"],
                d["we_win_spearman"], d["we_win_topK"]])
    net = "strong" if wins >= 4 else ("weak" if wins <= 2 else "mid")
    rows += (f'<tr class="posrow {net}"><th class="posname">{p}</th>'
             f'<td>{raw["ours_mae"]:.2f} <span class="sub">vs {raw["fp_mae"]:.2f}</span> {badge(raw["we_win_raw"])}</td>'
             f'<td>{d["points_ours"]["mae"]:.2f} <span class="sub">vs {d["points_fp"]["mae"]:.2f}</span> {badge(d["we_win_mae"])}</td>'
             f'<td>{d["points_ours"]["spearman"]:+.3f} <span class="sub">vs {d["points_fp"]["spearman"]:+.3f}</span> {badge(d["we_win_spearman"])}</td>'
             f'<td>{d["topK_ours"]:.2f} <span class="sub">vs {d["topK_fp"]:.2f}</span> {badge(d["we_win_topK"])}</td>'
             f'<td class="n">{d["n"]}</td></tr>')

# pooled stat cards
pr = pooled["raw_stat"]; pp = pooled["points"]; w = pooled["win_counts"]
cards = [
    ("pts MAE", f'{pp["ours"]["mae"]:.2f}', f'{pp["fp"]["mae"]:.2f}',
     pp["ours"]["mae"] < pp["fp"]["mae"], w["points_mae"]),
    ("pts RMSE", f'{pp["ours"]["rmse"]:.2f}', f'{pp["fp"]["rmse"]:.2f}',
     pp["ours"]["rmse"] < pp["fp"]["rmse"], w["points_rmse"]),
    ("Spearman", f'{pp["ours"]["spearman"]:+.3f}', f'{pp["fp"]["spearman"]:+.3f}',
     pp["ours"]["spearman"] > pp["fp"]["spearman"], w["spearman"]),
    (f"top{TOPK} decision", "3/4", "1/4",
     w[f"top{TOPK}_decision"] == "3/4", w[f"top{TOPK}_decision"]),
]
card_html = ""
for label, o, f, win, tally in cards:
    card_html += (f'<div class="card {("win" if win else "lose")}">'
                  f'<div class="card-lbl">{label}</div>'
                  f'<div class="card-val">{o}</div>'
                  f'<div class="card-fp">FP {f}</div>'
                  f'<div class="card-tally">{tally} positions</div>'
                  f'{badge(win)}</div>')

# weekly trend bars (Spearman)
maxrho = max(max(v["ours_rho"], v["fp_rho"], 0.1) for v in weeks.values())
wk_html = ""
for key in sorted(weeks):
    v = weeks[key]
    ow = abs(v["ours_rho"]) / maxrho * 100
    fw = abs(v["fp_rho"]) / maxrho * 100
    win = v["ours_rho"] > v["fp_rho"]
    wk_html += (
        f'<div class="wk"><div class="wk-lbl">{key}</div>'
        f'<div class="bars"><div class="bar ours" style="width:{ow:.0f}%"></div>'
        f'<span class="bar-val">{v["ours_rho"]:+.3f}</span></div>'
        f'<div class="bars"><div class="bar fp" style="width:{fw:.0f}%"></div>'
        f'<span class="bar-val">{v["fp_rho"]:+.3f}</span></div>'
        f'<div class="wk-win">{badge(win)}</div></div>')

weeks_list = ", ".join(f"w{w}" for w in rep["weeks"])

html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>2026 live head-to-head vs FantasyPros</title>
<style>
  :root {{ --accent:#38bdf8; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family: var(--font, system-ui, sans-serif); color: var(--foreground);
         margin:0; padding:14px 2px; }}
  h1 {{ font-size:17px; margin:0 0 2px; }}
  .sub-h {{ color: var(--muted-foreground); font-size:12.5px; margin:0 0 12px; }}
  .headline {{ background: var(--card); border:1px solid var(--border);
               border-left:3px solid var(--accent); border-radius:8px;
               padding:9px 12px; font-size:13.5px; margin-bottom:14px; }}
  .headline b {{ color: var(--accent); }}
  .cards {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px; }}
  .card {{ flex:1 1 120px; min-width:118px; background:var(--card);
           border:1px solid var(--border); border-radius:8px; padding:8px 10px; }}
  .card-lbl {{ font-size:11px; color:var(--muted-foreground); text-transform:uppercase; letter-spacing:.04em; }}
  .card-val {{ font-size:20px; font-weight:700; margin-top:2px; }}
  .card-fp {{ font-size:11.5px; color:var(--muted-foreground); }}
  .card-tally {{ font-size:11px; margin-top:3px; }}
  table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
  th,td {{ text-align:right; padding:5px 7px; border-bottom:1px solid var(--border); white-space:nowrap; }}
  th.posname, td.lbl {{ text-align:left; font-weight:600; }}
  thead th {{ color:var(--muted-foreground); font-size:11px; text-transform:uppercase; letter-spacing:.03em; }}
  .posrow.strong th.posname {{ color:var(--accent); }}
  .sub {{ color:var(--muted-foreground); font-size:11px; }}
  .n {{ color:var(--muted-foreground); font-size:11px; }}
  .badge {{ font-size:10px; font-weight:700; padding:1px 6px; border-radius:10px; letter-spacing:.03em; }}
  .badge.win {{ background:color-mix(in srgb, var(--accent) 22%, transparent); color:var(--accent); }}
  .badge.lose {{ background:color-mix(in srgb, var(--muted-foreground) 22%, transparent); color:var(--muted-foreground); }}
  .ours {{ color:var(--accent); font-weight:600; }}
  h2 {{ font-size:13px; margin:16px 0 6px; }}
  .wk {{ display:grid; grid-template-columns:34px 1fr 1fr 46px; gap:6px; align-items:center; margin:4px 0; }}
  .wk-lbl {{ font-size:11px; color:var(--muted-foreground); }}
  .bars {{ display:flex; align-items:center; gap:6px; }}
  .bar {{ height:9px; border-radius:5px; }}
  .bar.ours {{ background:var(--accent); }}
  .bar.fp {{ background:var(--muted-foreground); }}
  .bar-val {{ font-size:11px; }}
  .wk-win {{ text-align:right; }}
  .note {{ background:var(--card); border:1px solid var(--border); border-radius:8px;
           padding:9px 12px; font-size:11.5px; color:var(--muted-foreground); margin-top:14px; line-height:1.5; }}
  .legend {{ font-size:11px; color:var(--muted-foreground); margin:4px 0 8px; }}
  .legend .ours, .legend .fp {{ font-weight:600; }}
</style></head><body>
  <h1>2026 live head-to-head — our model vs the FantasyPros consensus</h1>
  <p class="sub-h">Scored live weeks {weeks_list} (the only true OOS for the shipped model) · ESPN Standard (non-PPR) · n={rep['n_player_weeks']} player-weeks</p>
  <div class="headline">We beat the consensus on the live 2026 weeks: pooled points <b>MAE 6.05 vs 6.22</b>,
  <b>Spearman +0.534 vs +0.512</b>, winning <b>3 of 4 positions</b> on points MAE/RMSE/Spearman.
  Modest but real — the edge is on signal/quality, not a blowout. WR is the one position we trail.</div>

  <div class="cards">{card_html}</div>

  <h2>By position</h2>
  <div class="legend"><span class="ours">ours</span> &nbsp;<span class="fp">vs FP</span> &nbsp;· rawMAE = apples-to-apples stat components · ptsRho = weekly points Spearman · top{TOPK}act = mean actual of each side's weekly top-{TOPK} pick</div>
  <table>
    <thead><tr><th>pos</th><th>raw-stat MAE</th><th>pts MAE</th><th>pts Spearman</th><th>top{TOPK} mean actual</th><th>n</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <h2>Weekly trend — points Spearman (ours vs FP)</h2>
  {wk_html}

  <div class="note"><b>Method:</b> apples-to-apples on the FantasyPros free-tier top-10/position board,
  compared against our logged pre-kickoff predictions (the clean live record) on the overlapping players;
  actuals from <code>features.parquet</code>. Both sides are pre-week commitments.
  <b>Caveats:</b> the FP free tier caps at top-10/position, so this is a top-10 head-to-head, not a full-roster
  one; single weeks are noisy (we win 6 of 9 on MAE, 6 of 9 on Spearman); WR is the one position where we trail
  on every metric, and QB top-{TOPK} is a near-tie. Repro: <code>uv run python notebooks/_fp_2026_scoreboard.py</code>.</div>
</body></html>"""

out = ROOT / "notebooks" / "fp_2026_scoreboard.html"
out.write_text(html_doc)
print(f"-> {out}  ({len(html_doc)} bytes)")
