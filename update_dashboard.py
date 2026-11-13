import os
from datetime import datetime, timezone
from pathlib import Path
import requests
import numpy as np
import pandas as pd

LEAGUE_ID = "809190147"
SEASON = 2026
REGULAR_SEASON_WEEKS = 14
PLAYOFF_SPOTS = 4
N_SIMULATIONS = 10000

COOKIES = {
    "swid": os.environ.get("SWID", ""),
    "espn_s2": os.environ.get("ESPN_S2", "")
}

URL = (
    f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{SEASON}"
    f"/segments/0/leagues/{LEAGUE_ID}?view=mTeam&view=mMatchup&view=mMatchupScore"
)

def build_dashboard():
    active_cookies = {k: v for k, v in COOKIES.items() if v}
    resp = requests.get(URL, cookies=active_cookies)
    resp.raise_for_status()
    data = resp.json()

    current_week = data["scoringPeriodId"]
    num_teams = len(data["teams"])

    teams = {
        t["id"]: f"{t.get('location', '')} {t.get('nickname', '')}".strip() or t.get("name", f"Team {t['id']}")
        for t in data["teams"]
    }

    games = []
    for m in data["schedule"]:
        week = m["matchupPeriodId"]
        if week > REGULAR_SEASON_WEEKS or "away" not in m:
            continue
        h_id, a_id = m["home"]["teamId"], m["away"]["teamId"]
        h_live = float(m["home"].get("totalPointsLive", m["home"].get("totalPoints", 0.0)))
        a_live = float(m["away"].get("totalPointsLive", m["away"].get("totalPoints", 0.0)))
        h_proj = float(m["home"].get("totalProjectedPointsLive", h_live))
        a_proj = float(m["away"].get("totalProjectedPointsLive", a_live))

        games.append({
            "week": week, "home_id": h_id, "away_id": a_id,
            "home_live": h_live, "away_live": a_live,
            "home_proj": h_proj, "away_proj": a_proj,
            "played": week <= current_week
        })

    sched_df = pd.DataFrame(games)
    weekly_rows = []
    for _, row in sched_df[sched_df["played"]].iterrows():
        w, h_id, a_id = row["week"], row["home_id"], row["away_id"]
        h_pts, a_pts = row["home_live"], row["away_live"]
        h_win = 1.0 if h_pts > a_pts else (0.5 if h_pts == a_pts else 0.0)
        weekly_rows.append({"week": w, "team_id": h_id, "team": teams[h_id], "pf": h_pts, "pa": a_pts, "actual_win": h_win})
        weekly_rows.append({"week": w, "team_id": a_id, "team": teams[a_id], "pf": a_pts, "pa": h_pts, "actual_win": 1.0 - h_win})

    weekly_df = pd.DataFrame(weekly_rows)

    prior_df = weekly_df[weekly_df["week"] < current_week]
    if not prior_df.empty:
        prior_summary = prior_df.groupby("team_id").agg(
            prior_games=("week", "count"), prior_wins=("actual_win", "sum"),
            prior_pf=("pf", "sum"), prior_pa=("pa", "sum"),
            hist_mean_pf=("pf", "mean"), hist_std_pf=("pf", "std")
        ).reset_index()
        prior_avg_pa = prior_summary["prior_pa"].mean()
        prior_summary["prior_exp"] = prior_summary["prior_games"] * (
            prior_summary["prior_pf"]**4 / (prior_summary["prior_pf"]**4 + prior_avg_pa**4)
        )
        prior_summary["prior_luck"] = prior_summary["prior_wins"] - prior_summary["prior_exp"]
    else:
        prior_summary = pd.DataFrame({"team_id": list(teams.keys()), "prior_luck": 0.0, "hist_mean_pf": 125.0, "hist_std_pf": 22.0})

    live_summary = weekly_df.groupby(["team_id", "team"]).agg(
        games=("week", "count"), wins=("actual_win", "sum"), pf=("pf", "sum"), pa=("pa", "sum")
    ).reset_index()
    live_summary["losses"] = live_summary["games"] - live_summary["wins"]
    live_avg_pa = live_summary["pa"].mean()
    live_summary["exp_wins"] = live_summary["games"] * (
        live_summary["pf"]**4 / (live_summary["pf"]**4 + live_avg_pa**4)
    )
    live_summary["live_luck"] = live_summary["wins"] - live_summary["exp_wins"]

    summary = pd.merge(
        live_summary, prior_summary[["team_id", "prior_luck", "hist_mean_pf", "hist_std_pf"]],
        on="team_id", how="left"
    ).fillna({"prior_luck": 0.0, "hist_mean_pf": 125.0, "hist_std_pf": 22.0})
    summary["luck_delta"] = summary["live_luck"] - summary["prior_luck"]

    curr_rows = []
    for _, row in sched_df[sched_df["week"] == current_week].iterrows():
        curr_rows.append({"team_id": row["home_id"], "wk_live": row["home_live"], "wk_opp_live": row["away_live"], "wk_proj": row["home_proj"], "wk_win": 1.0 if row["home_live"] > row["away_live"] else (0.5 if row["home_live"] == row["away_live"] else 0.0)})
        curr_rows.append({"team_id": row["away_id"], "wk_live": row["away_live"], "wk_opp_live": row["home_live"], "wk_proj": row["away_proj"], "wk_win": 1.0 if row["away_live"] > row["home_live"] else (0.5 if row["away_live"] == row["home_live"] else 0.0)})
    summary = pd.merge(summary, pd.DataFrame(curr_rows), on="team_id", how="left").fillna(0.0)

    team_ids = summary["team_id"].tolist()
    id_to_idx = {tid: i for i, tid in enumerate(team_ids)}

    if not prior_df.empty:
        base_df = summary.merge(prior_summary[["team_id", "prior_wins", "prior_pf"]], on="team_id")
        sim_wins = np.tile(base_df["prior_wins"].values.astype(float), (N_SIMULATIONS, 1))
        sim_pf = np.tile(base_df["prior_pf"].values.astype(float), (N_SIMULATIONS, 1))
    else:
        sim_wins = np.zeros((N_SIMULATIONS, num_teams))
        sim_pf = np.zeros((N_SIMULATIONS, num_teams))

    stds = summary["hist_std_pf"].replace(0, 22.0).values
    hist_means = summary["hist_mean_pf"].values
    curr_projs = summary["wk_proj"].values
    curr_lives = summary["wk_live"].values
    blended_future_means = 0.60 * hist_means + 0.40 * np.where(curr_projs > 0, curr_projs, hist_means)

    rng = np.random.default_rng(42)
    for _, row in sched_df[sched_df["week"] >= current_week].iterrows():
        h_idx, a_idx = id_to_idx[row["home_id"]], id_to_idx[row["away_id"]]
        if row["week"] == current_week:
            h_rem = max(0.0, (curr_projs[h_idx] - curr_lives[h_idx]) / max(curr_projs[h_idx], 1.0))
            a_rem = max(0.0, (curr_projs[a_idx] - curr_lives[a_idx]) / max(curr_projs[a_idx], 1.0))
            h_scores = np.maximum(rng.normal(curr_projs[h_idx], stds[h_idx] * np.sqrt(h_rem), N_SIMULATIONS), curr_lives[h_idx])
            a_scores = np.maximum(rng.normal(curr_projs[a_idx], stds[a_idx] * np.sqrt(a_rem), N_SIMULATIONS), curr_lives[a_idx])
        else:
            h_scores = rng.normal(blended_future_means[h_idx], stds[h_idx], N_SIMULATIONS)
            a_scores = rng.normal(blended_future_means[a_idx], stds[a_idx], N_SIMULATIONS)

        sim_pf[:, h_idx] += h_scores
        sim_pf[:, a_idx] += a_scores
        sim_wins[:, h_idx] += (h_scores > a_scores).astype(float)
        sim_wins[:, a_idx] += (a_scores > h_scores).astype(float)

    order = np.lexsort((-sim_pf, -sim_wins))
    ranks = np.argsort(order, axis=1)
    summary["proj_wins"] = sim_wins.mean(axis=0)
    summary["playoff_pct"] = (ranks < PLAYOFF_SPOTS).mean(axis=0) * 100
    summary["last_place_pct"] = (ranks == (num_teams - 1)).mean(axis=0) * 100
    summary = summary.sort_values(["playoff_pct", "wins", "pf"], ascending=[False, False, False]).reset_index(drop=True)

    def fmt_rec(w, l):
        return f"{int(w)}-{int(l)}" if w % 1 == 0 else f"{w:.1f}-{l:.1f}"

    def badge(val):
        color = "#15803d" if val > 0 else ("#b91c1c" if val < 0 else "#475569")
        bg = "#dcfce7" if val > 0 else ("#fee2e2" if val < 0 else "#f1f5f9")
        sign = "+" if val > 0 else ""
        return f'<span style="background:{bg};color:{color};padding:3px 8px;border-radius:6px;font-weight:600;">{sign}{val:.2f}</span>'

    rows_html = ""
    for idx, r in summary.iterrows():
        wk_color = "#15803d" if r["wk_win"] == 1 else ("#b91c1c" if r["wk_win"] == 0 else "#64748b")
        rows_html += f"""
        <tr>
            <td style="font-weight:600;">{idx + 1}</td>
            <td style="font-weight:700;color:#0f172a;text-align:left;">{r['team']}</td>
            <td>{fmt_rec(r['wins'], r['losses'])}</td>
            <td>{r['pf']:.2f}</td>
            <td>{r['pa']:.2f}</td>
            <td><span style="color:{wk_color};font-weight:600;">{r['wk_live']:.2f} - {r['wk_opp_live']:.2f}</span></td>
            <td style="font-weight:600;color:#334155;">{r['wk_proj']:.2f}</td>
            <td>{r['exp_wins']:.2f}</td>
            <td>{badge(r['live_luck'])}</td>
            <td>{badge(r['luck_delta'])}</td>
            <td style="font-weight:600;">{r['proj_wins']:.2f}</td>
            <td style="font-weight:700;color:#1d4ed8;">{r['playoff_pct']:.1f}%</td>
            <td style="font-weight:700;color:{'#b91c1c' if r['last_place_pct'] >= 15 else '#475569'};">{r['last_place_pct']:.1f}%</td>
        </tr>
        """

    timestamp = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="300">
  <title>EBIC 2026-7 Live Luck & Playoff Odds</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f1f5f9; color: #1e293b; margin: 0; padding: 16px; }}
    .card {{ max-width: 1150px; margin: 0 auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.06); padding: 20px; overflow-x: auto; }}
    h1 {{ margin: 0 0 6px 0; font-size: 20px; color: #0f172a; }}
    .sub {{ color: #64748b; font-size: 12px; margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: center; min-width: 850px; }}
    th {{ background: #0f172a; color: #fff; padding: 10px 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
    th:nth-child(2) {{ text-align: left; }}
    td {{ padding: 10px 8px; border-bottom: 1px solid #e2e8f0; }}
    tr:hover {{ background: #f8fafc; }}
    .footer {{ margin-top: 14px; font-size: 11px; color: #64748b; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>EBIC 2026-7 — Live Luck &amp; Playoff Projections (Week {current_week})</h1>
    <div class="sub">Last Synced: {timestamp} &bull; Auto-refreshes every 5 min &bull; {N_SIMULATIONS:,} Monte Carlo Sims (Top {PLAYOFF_SPOTS} Playoffs, PF Tiebreaker)</div>
    <table>
      <thead>
        <tr>
          <th>Rank</th><th>Team</th><th>Live Rec</th><th>Total PF</th><th>Total PA</th>
          <th>Wk {current_week} Live</th><th>Wk {current_week} Proj</th><th>Exp Wins</th>
          <th>Live Luck</th><th>Wk {current_week} Luck &Delta;</th><th>Proj Wins</th>
          <th>Playoff %</th><th>Last Place %</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div class="footer">
      <span><strong>Live Luck:</strong> Live Wins &minus; Pythagorean Expected Wins (k=4)</span>
      <span><strong>Wk {current_week} Luck &Delta;:</strong> Net luck added/removed since end of Week {max(1, current_week - 1)}</span>
    </div>
  </div>
</body>
</html>"""
    Path("index.html").write_text(html, encoding="utf-8")

if __name__ == "__main__":
    build_dashboard()
