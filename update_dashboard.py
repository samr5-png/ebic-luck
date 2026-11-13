import os
import json
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
    team_ids = sorted(teams.keys())
    id_to_idx = {tid: i for i, tid in enumerate(team_ids)}

    # --- 1. PARSE FULL SEASON SCHEDULE, LIVE POINTS & ESPN PROJECTIONS ---
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

    # --- 2. REPLAY SEASON WEEK-BY-WEEK WITH EMPIRICAL BAYES SHRINKAGE ---
    rng = np.random.default_rng(42)
    week_labels = []
    playoff_history = {tid: [] for tid in team_ids}
    last_place_history = {tid: [] for tid in team_ids}

    for w_eval in range(1, current_week + 1):
        is_live_week = (w_eval == current_week)
        cutoff = w_eval - 1 if is_live_week else w_eval
        past_games = sched_df[sched_df["week"] <= cutoff]

        wins = np.zeros(num_teams)
        pf = np.zeros(num_teams)
        scores_by_idx = {i: [] for i in range(num_teams)}
        all_past_scores = []

        for _, r in past_games.iterrows():
            h, a = id_to_idx[r["home_id"]], id_to_idx[r["away_id"]]
            hp, ap = r["home_live"], r["away_live"]
            pf[h] += hp
            pf[a] += ap
            scores_by_idx[h].append(hp)
            scores_by_idx[a].append(ap)
            all_past_scores.extend([hp, ap])
            if hp > ap:
                wins[h] += 1.0
            elif ap > hp:
                wins[a] += 1.0
            else:
                wins[h] += 0.5
                wins[a] += 0.5

        # League-wide scoring baseline for Bayesian regression to the mean
        league_mean = float(np.mean(all_past_scores)) if all_past_scores else 125.0
        league_var = float(np.var(all_past_scores, ddof=1)) if len(all_past_scores) > 1 else 23.0**2
        league_var = max(league_var, 18.0**2)

        # Shrink early-week team means (prior weight = 4 games) & variances (prior weight = 5 games)
        k_mean = 4.0
        k_var = 5.0
        means = np.zeros(num_teams)
        stds = np.zeros(num_teams)

        for i in range(num_teams):
            n = len(scores_by_idx[i])
            if n == 0:
                means[i] = league_mean
                stds[i] = np.sqrt(league_var)
            else:
                raw_mean = np.mean(scores_by_idx[i])
                means[i] = (n * raw_mean + k_mean * league_mean) / (n + k_mean)
                raw_var = np.var(scores_by_idx[i], ddof=1) if n > 1 else league_var
                shrunk_var = ((max(0, n - 1) * raw_var) + (k_var * league_var)) / (max(0, n - 1) + k_var)
                stds[i] = max(np.sqrt(shrunk_var), 12.0)

        curr_projs = np.zeros(num_teams)
        curr_lives = np.zeros(num_teams)
        if is_live_week:
            for _, r in sched_df[sched_df["week"] == current_week].iterrows():
                curr_projs[id_to_idx[r["home_id"]]] = r["home_proj"]
                curr_projs[id_to_idx[r["away_id"]]] = r["away_proj"]
                curr_lives[id_to_idx[r["home_id"]]] = r["home_live"]
                curr_lives[id_to_idx[r["away_id"]]] = r["away_live"]
            blended_means = 0.65 * means + 0.35 * np.where(curr_projs > 0, curr_projs, means)
        else:
            blended_means = means

        sim_w = np.tile(wins, (N_SIMULATIONS, 1))
        sim_pf = np.tile(pf, (N_SIMULATIONS, 1))

        for _, r in sched_df[sched_df["week"] > cutoff].iterrows():
            h, a = id_to_idx[r["home_id"]], id_to_idx[r["away_id"]]
            if is_live_week and r["week"] == current_week:
                h_rem = max(0.0, (curr_projs[h] - curr_lives[h]) / max(curr_projs[h], 1.0))
                a_rem = max(0.0, (curr_projs[a] - curr_lives[a]) / max(curr_projs[a], 1.0))
                hs = np.maximum(rng.normal(curr_projs[h], stds[h] * np.sqrt(h_rem), N_SIMULATIONS), curr_lives[h])
                as_ = np.maximum(rng.normal(curr_projs[a], stds[a] * np.sqrt(a_rem), N_SIMULATIONS), curr_lives[a])
            else:
                hs = rng.normal(blended_means[h], stds[h], N_SIMULATIONS)
                as_ = rng.normal(blended_means[a], stds[a], N_SIMULATIONS)

            sim_pf[:, h] += hs
            sim_pf[:, a] += as_
            sim_w[:, h] += (hs > as_).astype(float)
            sim_w[:, a] += (as_ > hs).astype(float)

        order = np.lexsort((-sim_pf, -sim_w))
        ranks = np.argsort(order, axis=1)
        playoff_pcts = (ranks < PLAYOFF_SPOTS).mean(axis=0) * 100
        last_pcts = (ranks == (num_teams - 1)).mean(axis=0) * 100

        week_labels.append(f"Wk {w_eval} (Live)" if is_live_week else f"Wk {w_eval}")
        for tid in team_ids:
            idx = id_to_idx[tid]
            playoff_history[tid].append(round(float(playoff_pcts[idx]), 1))
            last_place_history[tid].append(round(float(last_pcts[idx]), 1))

        if is_live_week:
            final_proj_wins = sim_w.mean(axis=0)
            final_playoff_pct = playoff_pcts
            final_last_pct = last_pcts

    # --- 3. COMPUTE LIVE LUCK & WEEKLY LUCK DELTA ---
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
            prior_pf=("pf", "sum"), prior_pa=("pa", "sum")
        ).reset_index()
        prior_avg_pa = prior_summary["prior_pa"].mean()
        prior_summary["prior_exp"] = prior_summary["prior_games"] * (
            prior_summary["prior_pf"]**4 / (prior_summary["prior_pf"]**4 + prior_avg_pa**4)
        )
        prior_summary["prior_luck"] = prior_summary["prior_wins"] - prior_summary["prior_exp"]
    else:
        prior_summary = pd.DataFrame({"team_id": team_ids, "prior_luck": 0.0})

    live_summary = weekly_df.groupby(["team_id", "team"]).agg(
        games=("week", "count"), wins=("actual_win", "sum"), pf=("pf", "sum"), pa=("pa", "sum")
    ).reset_index()
    live_summary["losses"] = live_summary["games"] - live_summary["wins"]
    live_avg_pa = live_summary["pa"].mean()
    live_summary["exp_wins"] = live_summary["games"] * (
        live_summary["pf"]**4 / (live_summary["pf"]**4 + live_avg_pa**4)
    )
    live_summary["live_luck"] = live_summary["wins"] - live_summary["exp_wins"]

    summary = pd.merge(live_summary, prior_summary[["team_id", "prior_luck"]], on="team_id", how="left").fillna({"prior_luck": 0.0})
    summary["luck_delta"] = summary["live_luck"] - summary["prior_luck"]

    curr_rows = []
    for _, row in sched_df[sched_df["week"] == current_week].iterrows():
        curr_rows.append({"team_id": row["home_id"], "wk_live": row["home_live"], "wk_opp_live": row["away_live"], "wk_proj": row["home_proj"], "wk_win": 1.0 if row["home_live"] > row["away_live"] else (0.5 if row["home_live"] == row["away_live"] else 0.0)})
        curr_rows.append({"team_id": row["away_id"], "wk_live": row["away_live"], "wk_opp_live": row["home_live"], "wk_proj": row["away_proj"], "wk_win": 1.0 if row["away_live"] > row["home_live"] else (0.5 if row["away_live"] == row["home_live"] else 0.0)})
    summary = pd.merge(summary, pd.DataFrame(curr_rows), on="team_id", how="left").fillna(0.0)

    summary["proj_wins"] = summary["team_id"].map(lambda tid: final_proj_wins[id_to_idx[tid]])
    summary["playoff_pct"] = summary["team_id"].map(lambda tid: final_playoff_pct[id_to_idx[tid]])
    summary["last_place_pct"] = summary["team_id"].map(lambda tid: final_last_pct[id_to_idx[tid]])
    summary = summary.sort_values(["playoff_pct", "wins", "pf"], ascending=[False, False, False]).reset_index(drop=True)

    # --- 4. PREPARE CHART.JS DATASETS ---
    palette = [
        "#2563eb", "#dc2626", "#16a34a", "#d97706",
        "#7c3aed", "#0891b2", "#db2777", "#475569"
    ]
    team_colors = {tid: palette[i % len(palette)] for i, tid in enumerate(team_ids)}

    playoff_datasets = []
    for _, r in summary.sort_values("playoff_pct", ascending=False).iterrows():
        tid = r["team_id"]
        playoff_datasets.append({
            "label": r["team"],
            "data": playoff_history[tid],
            "borderColor": team_colors[tid],
            "backgroundColor": team_colors[tid],
            "borderWidth": 2.5,
            "pointRadius": 4,
            "tension": 0.25,
            "fill": False
        })

    last_place_datasets = []
    for _, r in summary.sort_values("last_place_pct", ascending=False).iterrows():
        tid = r["team_id"]
        last_place_datasets.append({
            "label": r["team"],
            "data": last_place_history[tid],
            "borderColor": team_colors[tid],
            "backgroundColor": team_colors[tid],
            "borderWidth": 2.5,
            "pointRadius": 4,
            "tension": 0.25,
            "fill": False
        })

    # --- 5. BUILD HTML ---
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
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f1f5f9; color: #1e293b; margin: 0; padding: 16px; }}
    .card {{ max-width: 1150px; margin: 0 auto 20px auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.06); padding: 20px; overflow-x: auto; }}
    h1, h2 {{ margin: 0 0 6px 0; color: #0f172a; }}
    h1 {{ font-size: 20px; }}
    h2 {{ font-size: 17px; }}
    .sub {{ color: #64748b; font-size: 12px; margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: center; min-width: 850px; }}
    th {{ background: #0f172a; color: #fff; padding: 10px 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
    th:nth-child(2) {{ text-align: left; }}
    td {{ padding: 10px 8px; border-bottom: 1px solid #e2e8f0; }}
    tr:hover {{ background: #f8fafc; }}
    .footer {{ margin-top: 14px; font-size: 11px; color: #64748b; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }}
    .chart-box {{ position: relative; height: 380px; width: 100%; }}
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

  <div class="card">
    <h2>Playoff Race — Historical Playoff Odds Over the Season</h2>
    <div class="sub">Simulated probability of finishing Top {PLAYOFF_SPOTS} at the end of each week &bull; Click any team in the legend to toggle</div>
    <div class="chart-box">
      <canvas id="playoffChart"></canvas>
    </div>
  </div>

  <div class="card">
    <h2>Race for Last Place — Historical Odds Over the Season</h2>
    <div class="sub">Simulated probability of finishing {num_teams}th (Last Place) at the end of each week &bull; Click any team in the legend to toggle</div>
    <div class="chart-box">
      <canvas id="lastPlaceChart"></canvas>
    </div>
  </div>

  <script>
    const weekLabels = {json.dumps(week_labels)};

    function buildChart(canvasId, datasets, yAxisTitle) {{
      const ctx = document.getElementById(canvasId).getContext('2d');
      new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: weekLabels,
          datasets: datasets
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: 'index', intersect: false }},
          plugins: {{
            legend: {{ position: 'bottom', labels: {{ usePointStyle: true, padding: 16 }} }},
            tooltip: {{
              callbacks: {{
                label: function(context) {{
                  return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                }}
              }}
            }}
          }},
          scales: {{
            y: {{
              min: 0,
              max: 100,
              title: {{ display: true, text: yAxisTitle }},
              ticks: {{ callback: function(value) {{ return value + '%'; }} }}
            }}
          }}
        }}
      }});
    }}

    buildChart('playoffChart', {json.dumps(playoff_datasets)}, 'Playoff Probability (%)');
    buildChart('lastPlaceChart', {json.dumps(last_place_datasets)}, 'Last-Place Probability (%)');
  </script>
</body>
</html>"""
    Path("index.html").write_text(html, encoding="utf-8")

if __name__ == "__main__":
    build_dashboard()
