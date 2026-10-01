import polars as pl, json
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
s = pl.read_parquet(REPO / "data/raw/schedules.parquet").filter((pl.col("season") == 2026) & (pl.col("week") == 4))
s = s.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in ["spread_line", "total_line"]])
s = s.with_columns(((pl.col("total_line") - pl.col("spread_line")) / 2).alias("a"),
                   ((pl.col("total_line") + pl.col("spread_line")) / 2).alias("h"))
rows = []
for g in s.to_dicts():
    rows.append(dict(team=g["away_team"], opp=g["home_team"], proj_allwd=round(g["h"], 1)))
    rows.append(dict(team=g["home_team"], opp=g["away_team"], proj_allwd=round(g["a"], 1)))
def tier(a):
    if a <= 6: return 10
    if a <= 13: return 7
    if a <= 17: return 4
    if a <= 23: return 1
    if a <= 30: return 0
    return -1
SACKS, TO = 2.36, 1.03
for r in rows:
    r["allowed_pts"] = tier(r["proj_allwd"])
    r["d_st"] = round(r["allowed_pts"] + SACKS + 2 * TO, 1)
rows.sort(key=lambda x: (-x["d_st"], x["proj_allwd"]))
for i, r in enumerate(rows, 1):
    r["rank"] = i
(REPO / "notebooks/_defboard_data.json").write_text(json.dumps(rows), encoding="utf-8")
print("wrote", len(rows), "rows; top:", [r["team"] for r in rows[:5]])
