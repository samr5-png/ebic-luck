import os
import json
import math
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

BASE_URL = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}"


def _sum_entries(entries, current_week):
    total_proj = 0.0
    for entry in entries:
        # 20 = Bench, 21 = IR; only sum active starters
        if entry.get("lineupSlotId") in (20, 21):
            continue
        ppe = entry.get("playerPoolEntry", {})
        player = ppe.get("player", {})

        actual_pts = None
        proj_pts = None
        for st in player.get("stats", []):
            if st.get("scoringPeriodId") == current_week:
                if st.get("statSourceId") == 0:
                    actual_pts = float(st.get("appliedTotal", 0.0))
                elif st.get("statSourceId") == 1:
                    proj_pts = float(st.get("appliedTotal", 0.0))

        # ESPN dynamically updates statSourceId == 1 during live games (e.g., 12.42 live -> 24.8 live proj).
        # Once a game finishes, statSourceId == 1 equals final actual points (or use max of actual & proj).
        if actual_pts is not None and proj_pts is not None:
            total_proj += max(actual_pts, proj_pts) if proj_pts > 0 else actual_pts
        elif proj_pts is not None:
            total_proj += proj_pts
        elif actual_pts is not None:
            total_proj += actual_pts
        else:
            total_proj += float(ppe.get("appliedStatTotal", 0.0))

    return round(total_proj, 2)


def get_starting_lineup_projections(data, current_week):
    roster_projs = {}
    for m in data.get("schedule", []):
        if m.get("matchupPeriodId") != current_week:
            continue
        for side in ("home", "away"):
            if side not in m:
                continue
            tid = m[side]["teamId"]
            entries = m[side].get("rosterForCurrentScoringPeriod", {}).get("entries", [])
            if entries:
                roster_projs[tid] = _sum_entries(entries, current_week)

    for t in data.get("teams", []):
        tid = t["id"]
        if tid in roster_projs and roster_projs[tid] > 0:
            continue
        entries = t.get("roster", {}).get("entries", [])
        roster_projs[tid] = _sum_entries(entries, current_week)

    return roster_projs


def build_master_site():
    active_cookies = {k: v for k, v in COOKIES.items() if v}

    # --- 1. FETCH CURRENT WEEK & LIVE ROSTERS/BOXSCORES ---
    meta_resp = requests.get(BASE_URL, cookies=active_cookies)
    meta_resp.raise_for_status()
    current_week = meta_resp.json()["scoringPeriodId"]

    full_url = f"{BASE_URL}?scoringPeriodId={current_week}&view=mTeam&view=mMatchup&view=mMatchupScore&view=mRoster&view=mBoxscore"
    resp = requests.get(full_url, cookies=active_cookies)
    resp.raise_for_status()
    data = resp.json()

    num_teams = len(data["teams"])
    teams = {
        t["id"]: f"{t.get('location', '')} {t.get('nickname', '')}".strip() or t.get("name", f"Team {t['id']}")
        for t in data["teams"]
    }
    team_ids = sorted(teams.keys())
    id_to_idx = {tid: i for i, tid in enumerate(team_ids)}

    roster_projs = get_starting_lineup_projections(data, current_week)

    # --- 2. PARSE SCHEDULE & SIMULATE LUCK / PLAYOFF / LAST-PLACE ODDS ---
    games = []
    for m in data["schedule"]:
        week = m["matchupPeriodId"]
        if week > REGULAR_SEASON_WEEKS or "away" not in m:
            continue
        h_id, a_id = m["home"]["teamId"], m["away"]["teamId"]
        h_live = float(m["home"].get("totalPointsLive", m["home"].get("totalPoints", 0.0)))
        a_live = float(m["away"].get("totalPointsLive", m["away"].get("totalPoints", 0.0)))

        # Use ESPN's live dynamic projection (totalProjectedPointsLive) when active; fallback to starter sum
        h_live_proj = float(m["home"].get("totalProjectedPointsLive") or 0.0)
        a_live_proj = float(m["away"].get("totalProjectedPointsLive") or 0.0)
        h_proj = h_live_proj if h_live_proj > 0 else float(roster_projs.get(h_id) or h_live)
        a_proj = a_live_proj if a_live_proj > 0 else float(roster_projs.get(a_id) or a_live)

        games.append({
            "week": week, "home_id": h_id, "away_id": a_id,
            "home_live": h_live, "away_live": a_live,
            "home_proj": h_proj, "away_proj": a_proj,
            "played": week <= current_week
        })

    sched_df = pd.DataFrame(games)

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

        league_mean = float(np.mean(all_past_scores)) if all_past_scores else 125.0
        league_var = max(float(np.var(all_past_scores, ddof=1)) if len(all_past_scores) > 1 else 23.0**2, 18.0**2)

        k_mean, k_var = 4.0, 5.0
        means, stds = np.zeros(num_teams), np.zeros(num_teams)
        for i in range(num_teams):
            n = len(scores_by_idx[i])
            if n == 0:
                means[i], stds[i] = league_mean, np.sqrt(league_var)
            else:
                raw_mean = np.mean(scores_by_idx[i])
                means[i] = (n * raw_mean + k_mean * league_mean) / (n + k_mean)
                raw_var = np.var(scores_by_idx[i], ddof=1) if n > 1 else league_var
                shrunk_var = ((max(0, n - 1) * raw_var) + (k_var * league_var)) / (max(0, n - 1) + k_var)
                stds[i] = max(np.sqrt(shrunk_var), 12.0)

        curr_projs, curr_lives = np.zeros(num_teams), np.zeros(num_teams)
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

    palette = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#475569"]
    team_colors = {tid: palette[i % len(palette)] for i, tid in enumerate(team_ids)}

    playoff_datasets = [
        {"label": r["team"], "data": playoff_history[r["team_id"]], "borderColor": team_colors[r["team_id"]], "backgroundColor": team_colors[r["team_id"]], "borderWidth": 2.5, "pointRadius": 4, "tension": 0.25, "fill": False}
        for _, r in summary.sort_values("playoff_pct", ascending=False).iterrows()
    ]
    last_place_datasets = [
        {"label": r["team"], "data": last_place_history[r["team_id"]], "borderColor": team_colors[r["team_id"]], "backgroundColor": team_colors[r["team_id"]], "borderWidth": 2.5, "pointRadius": 4, "tension": 0.25, "fill": False}
        for _, r in summary.sort_values("last_place_pct", ascending=False).iterrows()
    ]

    # --- 3. FETCH HISTORICAL BOXSCORES & TRANSACTIONS FOR TABS 2 & 3 ---
    weekly_pts = {}
    started_set = set()
    player_names = {}

    for wk in range(1, current_week + 1):
        box_resp = requests.get(f"{BASE_URL}?scoringPeriodId={wk}&view=mBoxscore", cookies=active_cookies)
        if box_resp.status_code != 200:
            continue
        box_data = box_resp.json()
        for m in box_data.get("schedule", []):
            if m.get("matchupPeriodId") != wk:
                continue
            for side in ("home", "away"):
                if side not in m:
                    continue
                tid = m[side]["teamId"]
                for entry in m[side].get("rosterForCurrentScoringPeriod", {}).get("entries", []):
                    is_starter = entry.get("lineupSlotId") not in (20, 21)
                    ppe = entry.get("playerPoolEntry", {})
                    p = ppe.get("player", {})
                    pid = p.get("id") or entry.get("playerId")
                    if not pid:
                        continue
                    if p.get("fullName"):
                        player_names[pid] = p["fullName"]
                    if is_starter:
                        started_set.add((tid, wk, pid))
                    pts_val = ppe.get("appliedStatTotal")
                    if pts_val is None:
                        for st in p.get("stats", []):
                            if st.get("scoringPeriodId") == wk and st.get("statSourceId") == 0:
                                pts_val = st.get("appliedTotal", 0.0)
                                break
                    weekly_pts.setdefault(pid, {})[wk] = float(pts_val or 0.0)

    headers = {"X-Fantasy-Filter": json.dumps({"players": {"limit": 1500, "sortPercOwned": {"sortAsc": False, "sortPriority": 1}}})}
    card_resp = requests.get(f"{BASE_URL}?view=kona_playercard", headers=headers, cookies=active_cookies)
    card_resp.raise_for_status()
    card_data = card_resp.json()

    player_movements = {}
    grouped_trades = {}
    pickups = []
    seen_pickup_keys = set()

    for p_entry in card_data.get("players", []):
        player = p_entry.get("player", {})
        pid = player.get("id")
        if not pid:
            continue
        player_names[pid] = player.get("fullName") or player_names.get(pid, f"Player #{pid}")

        seen_tx = set()
        for tx in p_entry.get("transactions", []):
            if tx.get("status") != "EXECUTED":
                continue
            tx_type = tx.get("type")
            tx_id = tx.get("relatedTransactionId") or tx.get("id")
            week = int(tx.get("scoringPeriodId", 1))
            ts_ms = int(tx.get("processDate") or tx.get("proposedDate") or (week * 1000000))
            bid_amt = tx.get("bidAmount")

            for item in tx.get("items", []):
                if item.get("playerId") != pid:
                    continue
                f_id = item.get("fromTeamId") or 0
                t_id = item.get("toTeamId") or 0
                item_type = item.get("type", "")
                key = (tx_id, f_id, t_id, item_type)
                if key in seen_tx:
                    continue
                seen_tx.add(key)

                player_movements.setdefault(pid, []).append((ts_ms, week, f_id, t_id, tx_type, tx_id))

                if tx_type == "TRADE_ACCEPT" and f_id and t_id:
                    if tx_id not in grouped_trades:
                        grouped_trades[tx_id] = {
                            "id": tx_id, "week": week, "ts": ts_ms,
                            "date": datetime.fromtimestamp(ts_ms / 1000).strftime("%b %d") if ts_ms > 100000000 else f"Wk {week}",
                            "legs": []
                        }
                    leg = (f_id, t_id, pid)
                    if leg not in grouped_trades[tx_id]["legs"]:
                        grouped_trades[tx_id]["legs"].append(leg)

                elif tx_type in ("WAIVER", "FREEAGENT") and t_id in teams and (item_type == "ADD" or f_id == 0):
                    p_key = (tx_id, pid, t_id)
                    if p_key not in seen_pickup_keys:
                        seen_pickup_keys.add(p_key)
                        pickups.append({
                            "tx_id": tx_id, "pid": pid, "team_id": t_id, "week": week, "ts": ts_ms,
                            "method": "Waiver" if tx_type == "WAIVER" else "Free Agent", "bid": bid_amt,
                            "date": datetime.fromtimestamp(ts_ms / 1000).strftime("%b %d") if ts_ms > 100000000 else f"Wk {week}"
                        })

    for pid in player_movements:
        player_movements[pid].sort(key=lambda x: (x[0], x[1]))

    def get_started_tenure_points(pid, team_id, start_ts, start_week):
        moves = player_movements.get(pid, [])
        end_week = current_week
        status_note = ""
        for m_ts, m_wk, f_id, t_id, m_type, _ in moves:
            if m_ts > start_ts and f_id == team_id:
                end_week = m_wk
                status_note = f", traded Wk {m_wk}" if m_type == "TRADE_ACCEPT" else f", dropped Wk {m_wk}"
                break

        p_map = weekly_pts.get(pid, {})
        pts, starts = 0.0, 0
        for w in range(start_week, end_week + 1):
            if (team_id, w, pid) in started_set:
                pts += p_map.get(w, 0.0)
                starts += 1

        start_label = f"{starts} start{'s' if starts != 1 else ''}{status_note}"
        return round(pts, 1), starts, f" ({start_label})"

    team_trade_pts = {tid: 0.0 for tid in teams}
    team_trade_count = {tid: 0 for tid in teams}
    graded_trades = []

    pair_counts = {}
    pair_details = {}
    player_trade_chains = {}

    for tr in sorted(grouped_trades.values(), key=lambda x: (x["week"], x["ts"]), reverse=True):
        wk, ts = tr["week"], tr["ts"]
        involved_names = {player_names.get(pid, "") for _, _, pid in tr["legs"]}
        # Exclude Week 8 forced trade (Amon-Ra St. Brown <-> Zay Flowers)
        if wk == 8 and "Amon-Ra St. Brown" in involved_names and "Zay Flowers" in involved_names:
            continue

        team_acquired = {}
        for f_id, t_id, pid in tr["legs"]:
            pts, _, note = get_started_tenure_points(pid, t_id, ts, wk)
            team_acquired.setdefault(t_id, []).append((f"{player_names.get(pid, f'#{pid}')}{note}", pts))
            team_acquired.setdefault(f_id, [])
            player_trade_chains.setdefault(pid, []).append((ts, wk, f_id, t_id))

        involved = list(team_acquired.keys())
        if len(involved) != 2:
            continue

        t1, t2 = involved[0], involved[1]
        pair_key = (min(t1, t2), max(t1, t2))
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        pair_details.setdefault(pair_key, []).append(f"Wk {wk}")

        t1_total = round(sum(p for _, p in team_acquired[t1]), 1)
        t2_total = round(sum(p for _, p in team_acquired[t2]), 1)
        diff = round(t1_total - t2_total, 1)

        team_trade_pts[t1] += diff
        team_trade_pts[t2] -= diff
        team_trade_count[t1] += 1
        team_trade_count[t2] += 1

        graded_trades.append({
            "kind": "TRADE", "week": wk, "ts": ts, "date": tr["date"],
            "t1_id": t1, "t1_name": teams.get(t1, "Team 1"), "t1_assets": team_acquired[t1], "t1_total": t1_total,
            "t2_id": t2, "t2_name": teams.get(t2, "Team 2"), "t2_assets": team_acquired[t2], "t2_total": t2_total,
            "impact": abs(diff), "winner_id": t1 if diff > 0 else (t2 if diff < 0 else None)
        })

    # Clean circular-layout Vis.js nodes & aggregated pair edges for Tab 3
    radius = 235
    ordered_web_ids = sorted(team_ids, key=lambda tid: team_trade_count[tid], reverse=True)
    ring_order = []
    left, right = 0, len(ordered_web_ids) - 1
    while left <= right:
        ring_order.append(ordered_web_ids[left])
        if left != right:
            ring_order.append(ordered_web_ids[right])
        left += 1
        right -= 1

    web_nodes = []
    for idx, tid in enumerate(ring_order):
        angle = (2 * math.pi * idx) / len(ring_order) - (math.pi / 2)
        cnt = team_trade_count[tid]
        web_nodes.append({
            "id": tid,
            "x": round(radius * math.cos(angle), 1),
            "y": round(radius * math.sin(angle), 1),
            "label": f"{teams[tid]}\n({cnt} trade{'s' if cnt != 1 else ''})",
            "size": 18 + (cnt * 1.5),
            "color": {"background": team_colors[tid], "border": "#0f172a"},
            "font": {"color": "#0f172a", "size": 13, "face": "sans-serif", "bold": True}
        })

    web_edges = []
    for (u, v), cnt in pair_counts.items():
        wks_str = ", ".join(reversed(pair_details[(u, v)]))
        web_edges.append({
            "from": u,
            "to": v,
            "width": 1.5 + (cnt * 1.5),
            "label": f"{cnt}x",
            "title": f"{teams[u]} ↔ {teams[v]}: {cnt} trade{'s' if cnt != 1 else ''} ({wks_str})",
            "font": {
                "size": 12,
                "align": "horizontal",
                "background": "#ffffff",
                "color": "#0f172a",
                "strokeWidth": 3,
                "strokeColor": "#ffffff"
            },
            "color": {"color": "#94a3b8", "highlight": "#2563eb", "hover": "#2563eb"}
        })

    # Most-Traded Players ("Hot Potatoes") table
    hot_potato_list = []
    for pid, chain in player_trade_chains.items():
        chain.sort(key=lambda x: (x[0], x[1]))
        count = len(chain)
        route_parts = [f"<strong>{teams.get(chain[0][2], 'Team')}</strong>"]
        for _, w, _, to_id in chain:
            route_parts.append(f"<span style='color:#64748b;font-size:11px;'>&rarr; (Wk {w})</span> <strong>{teams.get(to_id, 'Team')}</strong>")
        hot_potato_list.append({
            "name": player_names.get(pid, f"Player #{pid}"),
            "count": count,
            "route_html": " ".join(route_parts)
        })

    hot_potato_list.sort(key=lambda x: (-x["count"], x["name"]))
    hot_potato_rows = ""
    for hp in hot_potato_list:
        if hp["count"] < 2:
            continue
        badge_bg = "#fee2e2" if hp["count"] >= 3 else "#fef3c7"
        badge_col = "#b91c1c" if hp["count"] >= 3 else "#b45309"
        hot_potato_rows += f"""
        <tr>
          <td style="font-weight:700;color:#0f172a;text-align:left;">{hp['name']}</td>
          <td><span style="background:{badge_bg};color:{badge_col};padding:3px 10px;border-radius:999px;font-weight:700;font-size:12px;">{hp['count']}x Traded</span></td>
          <td style="text-align:left;line-height:1.6;">{hp['route_html']}</td>
        </tr>
        """

    team_waiver_pts = {tid: 0.0 for tid in teams}
    team_pickup_count = {tid: 0 for tid in teams}
    evaluated_pickups = []

    for pk in sorted(pickups, key=lambda x: (x["week"], x["ts"]), reverse=True):
        pid, tid = pk["pid"], pk["team_id"]
        pts, starts, note = get_started_tenure_points(pid, tid, pk["ts"], pk["week"])
        team_waiver_pts[tid] += pts
        team_pickup_count[tid] += 1

        bid_str = f" (${pk['bid']} FAAB)" if pk["method"] == "Waiver" and pk["bid"] is not None and pk["bid"] > 0 else ""
        evaluated_pickups.append({
            "kind": "PICKUP", "week": pk["week"], "ts": pk["ts"], "date": pk["date"],
            "team_id": tid, "team_name": teams.get(tid, "Team"),
            "player_name": player_names.get(pid, f"Player #{pid}"),
            "method": f"{pk['method']}{bid_str}", "impact": pts, "starts": starts, "note": note.strip(" ()")
        })

    team_total_pts = {tid: round(team_trade_pts[tid] + team_waiver_pts[tid], 1) for tid in teams}
    sorted_tx_teams = sorted(teams.keys(), key=lambda tid: team_total_pts[tid], reverse=True)

    tx_bar_labels = [f"{teams[tid]} ({team_trade_count[tid]}T / {team_pickup_count[tid]}W)" for tid in sorted_tx_teams]
    tx_trade_series = [round(team_trade_pts[tid], 1) for tid in sorted_tx_teams]
    tx_waiver_series = [round(team_waiver_pts[tid], 1) for tid in sorted_tx_teams]

    team_options = '<option value="ALL">All Managers (Show Entire League)</option>' + "".join(
        f'<option value="{tid}">{teams[tid]} ({team_total_pts[tid]:+.1f} total pts)</option>'
        for tid in sorted_tx_teams
    )

    # --- 4. RENDER COMBINED HTML WITH THREE TOP TABS ---
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

    unified_items = sorted(graded_trades + evaluated_pickups, key=lambda x: (x["week"], x["ts"]), reverse=True)
    feed_html = ""
    for item in unified_items:
        if item["kind"] == "TRADE":
            t1_win = item["winner_id"] == item["t1_id"]
            t2_win = item["winner_id"] == item["t2_id"]
            diff_str = f"+{item['impact']:.1f} pts" if item["impact"] > 0 else "Even (0.0 pts)"

            t1_list = "".join(
                f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px dashed #e2e8f0;font-size:13px;'>"
                f"<span>{name}</span><span style='font-weight:600;white-space:nowrap;margin-left:8px;'>{pts:.1f} pts</span></div>"
                for name, pts in item["t1_assets"]
            ) or "<div style='color:#94a3b8;font-size:12px;'>No players recorded</div>"

            t2_list = "".join(
                f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px dashed #e2e8f0;font-size:13px;'>"
                f"<span>{name}</span><span style='font-weight:600;white-space:nowrap;margin-left:8px;'>{pts:.1f} pts</span></div>"
                for name, pts in item["t2_assets"]
            ) or "<div style='color:#94a3b8;font-size:12px;'>No players recorded</div>"

            badge_t1 = f"<span style='background:#dcfce7;color:#15803d;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;'>WINNER ({diff_str})</span>" if t1_win else ""
            badge_t2 = f"<span style='background:#dcfce7;color:#15803d;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;'>WINNER ({diff_str})</span>" if t2_win else ""

            feed_html += f"""
            <div class="tx-card" data-kind="TRADE" data-t1="{item['t1_id']}" data-t2="{item['t2_id']}" data-impact="{item['impact']}" data-ts="{item['ts']}">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #e2e8f0;">
                <div style="display:flex;align-items:center;gap:8px;">
                  <span style="background:#fef3c7;color:#b45309;padding:2px 8px;border-radius:6px;font-size:11px;font-weight:700;">TRADE</span>
                  <span style="font-size:12px;font-weight:700;color:#475569;">WEEK {item['week']} &bull; {item['date']}</span>
                </div>
                <span style="font-size:12px;font-weight:600;color:#64748b;">Margin: {diff_str}</span>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                <div style="background:{'#f0fdf4' if t1_win else '#f8fafc'};border:1px solid {'#bbf7d0' if t1_win else '#e2e8f0'};border-radius:8px;padding:12px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:6px;">
                    <strong style="font-size:14px;color:#0f172a;">{item['t1_name']} Received</strong>
                    {badge_t1}
                  </div>
                  {t1_list}
                  <div style="display:flex;justify-content:space-between;margin-top:8px;font-weight:700;color:#0f172a;font-size:13px;">
                    <span>Starter Points</span><span>{item['t1_total']:.1f} pts</span>
                  </div>
                </div>
                <div style="background:{'#f0fdf4' if t2_win else '#f8fafc'};border:1px solid {'#bbf7d0' if t2_win else '#e2e8f0'};border-radius:8px;padding:12px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:6px;">
                    <strong style="font-size:14px;color:#0f172a;">{item['t2_name']} Received</strong>
                    {badge_t2}
                  </div>
                  {t2_list}
                  <div style="display:flex;justify-content:space-between;margin-top:8px;font-weight:700;color:#0f172a;font-size:13px;">
                    <span>Starter Points</span><span>{item['t2_total']:.1f} pts</span>
                  </div>
                </div>
              </div>
            </div>
            """
        else:
            pts_color = "#15803d" if item["impact"] >= 20 else ("#1d4ed8" if item["impact"] > 0 else "#64748b")
            badge_bg = "#eff6ff" if "Waiver" in item["method"] else "#f1f5f9"
            badge_color = "#1d4ed8" if "Waiver" in item["method"] else "#475569"

            feed_html += f"""
            <div class="tx-card" data-kind="PICKUP" data-t1="{item['team_id']}" data-t2="" data-impact="{item['impact']}" data-ts="{item['ts']}">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                  <span style="background:{badge_bg};color:{badge_color};padding:2px 8px;border-radius:6px;font-size:11px;font-weight:700;text-transform:uppercase;">{item['method']}</span>
                  <span style="font-size:12px;font-weight:700;color:#475569;">WEEK {item['week']} ({item['date']})</span>
                  <span style="font-size:14px;color:#0f172a;"><strong>{item['team_name']}</strong> added <strong>{item['player_name']}</strong></span>
                  <span style="font-size:12px;color:#64748b;">({item['note']})</span>
                </div>
                <div style="font-size:15px;font-weight:700;color:{pts_color};">
                  +{item['impact']:.1f} starter pts
                </div>
              </div>
            </div>
            """

    timestamp = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="300">
  <title>EBIC 2026-7 Analytics Hub</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f1f5f9; color: #1e293b; margin: 0; padding: 0; }}
    .navbar {{ background: #0f172a; color: #fff; padding: 14px 20px; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 10px rgba(0,0,0,0.15); }}
    .nav-inner {{ max-width: 1150px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
    .brand {{ font-size: 17px; font-weight: 800; letter-spacing: -0.01em; }}
    .nav-tabs {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .main-tab {{ padding: 8px 16px; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: #cbd5e1; font-size: 13px; font-weight: 700; cursor: pointer; transition: all 0.15s; }}
    .main-tab:hover {{ background: #334155; color: #fff; }}
    .main-tab.active {{ background: #2563eb; color: #ffffff; border-color: #2563eb; }}
    .page-wrap {{ padding: 18px 16px; }}
    .card {{ max-width: 1150px; margin: 0 auto 20px auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.06); padding: 20px; overflow-x: auto; }}
    h1, h2 {{ margin: 0 0 6px 0; color: #0f172a; }}
    h1 {{ font-size: 20px; }}
    h2 {{ font-size: 17px; }}
    .sub {{ color: #64748b; font-size: 12px; margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: center; min-width: 650px; }}
    th {{ background: #0f172a; color: #fff; padding: 10px 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
    th:nth-child(2) {{ text-align: left; }}
    td {{ padding: 10px 8px; border-bottom: 1px solid #e2e8f0; }}
    tr:hover {{ background: #f8fafc; }}
    .footer {{ margin-top: 14px; font-size: 11px; color: #64748b; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }}
    .chart-box {{ position: relative; height: 360px; width: 100%; }}
    .tx-card {{ border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; margin-bottom: 12px; background: #ffffff; }}
    .sub-tab-bar {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .sub-tab {{ padding: 7px 13px; border-radius: 8px; border: 1px solid #cbd5e1; background: #f8fafc; color: #475569; font-size: 12px; font-weight: 700; cursor: pointer; }}
    .sub-tab.active {{ background: #0f172a; color: #ffffff; border-color: #0f172a; }}
    select {{ padding: 8px 12px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 13px; font-weight: 600; color: #0f172a; background: #f8fafc; }}
    #tradeWebNet {{ width: 100%; height: 540px; border: 1px solid #e2e8f0; border-radius: 10px; background: #fafafa; }}
  </style>
</head>
<body>
  <div class="navbar">
    <div class="nav-inner">
      <div class="brand">EBIC 2026-7 &bull; League Analytics</div>
      <div class="nav-tabs">
        <button class="main-tab active" onclick="switchPage('luckPage', this)">Luck &amp; Playoff Odds</button>
        <button class="main-tab" onclick="switchPage('txPage', this)">Trades &amp; Waiver Wire</button>
        <button class="main-tab" onclick="switchPage('webPage', this)">Trade Web &amp; Hot Potatoes</button>
      </div>
    </div>
  </div>

  <div class="page-wrap">
    <!-- TAB 1: LUCK & PLAYOFF ODDS -->
    <div id="luckPage">
      <div class="card">
        <h1>Live Luck &amp; Playoff Projections (Week {current_week})</h1>
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
        <div class="chart-box"><canvas id="playoffChart"></canvas></div>
      </div>

      <div class="card">
        <h2>Race for Last Place — Historical Odds Over the Season</h2>
        <div class="sub">Simulated probability of finishing {num_teams}th (Last Place) at the end of each week &bull; Click any team in the legend to toggle</div>
        <div class="chart-box"><canvas id="lastPlaceChart"></canvas></div>
      </div>
    </div>

    <!-- TAB 2: TRADES & WAIVER WIRE -->
    <div id="txPage" style="display:none;">
      <div class="card">
        <h1>Combined Trade &amp; Waiver Wire Value ({len(graded_trades)} Trades, {len(evaluated_pickups)} Pickups)</h1>
        <div class="sub">Net Trade Starter Points + Waiver/FA Starter Points scored <strong>only while started</strong> before being dropped or re-traded (Weeks 1–{current_week})</div>
        <div style="height: 330px; width: 100%;"><canvas id="combinedTxChart"></canvas></div>
      </div>

      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:16px;">
          <div class="sub-tab-bar">
            <button class="sub-tab active" onclick="setKind('ALL', this)">All Moves ({len(unified_items)})</button>
            <button class="sub-tab" onclick="setKind('TRADE', this)">Trades Only ({len(graded_trades)})</button>
            <button class="sub-tab" onclick="setKind('PICKUP', this)">Waivers &amp; FA Only ({len(evaluated_pickups)})</button>
          </div>
          <div style="display:flex;gap:10px;flex-wrap:wrap;">
            <select id="teamFilter" onchange="applyFilters()">{team_options}</select>
            <select id="sortOrder" onchange="applyFilters()">
              <option value="recent">Sort: Most Recent</option>
              <option value="impact">Sort: Biggest Point Impact</option>
            </select>
          </div>
        </div>
        <div id="feedContainer">{feed_html}</div>
      </div>
    </div>

    <!-- TAB 3: TRADE WEB & MOST TRADED PLAYERS -->
    <div id="webPage" style="display:none;">
      <div class="card">
        <h1>Manager Trade Partner Web ({len(graded_trades)} Trades)</h1>
        <div class="sub">Line thickness and badges show total trades between each pair &bull; Hover over any line to see the exact weeks they traded</div>
        <div id="tradeWebNet"></div>
      </div>

      <div class="card">
        <h2>Most-Traded Players ("Hot Potatoes")</h2>
        <div class="sub">Players dealt 2 or more times this season with their chronological ownership path</div>
        <table>
          <thead>
            <tr>
              <th style="text-align:left;">Player</th>
              <th>Times Traded</th>
              <th style="text-align:left;">Ownership Chain</th>
            </tr>
          </thead>
          <tbody>{hot_potato_rows or '<tr><td colspan="3" style="color:#64748b;">No players traded multiple times yet.</td></tr>'}</tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    let webInitialized = false;
    function switchPage(pageId, btn) {{
      document.getElementById('luckPage').style.display = (pageId === 'luckPage') ? 'block' : 'none';
      document.getElementById('txPage').style.display = (pageId === 'txPage') ? 'block' : 'none';
      document.getElementById('webPage').style.display = (pageId === 'webPage') ? 'block' : 'none';
      document.querySelectorAll('.main-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      if (pageId === 'webPage' && !webInitialized) {{
        webInitialized = true;
        const container = document.getElementById('tradeWebNet');
        const netData = {{
          nodes: new vis.DataSet({json.dumps(web_nodes)}),
          edges: new vis.DataSet({json.dumps(web_edges)})
        }};
        const netOptions = {{
          nodes: {{ shape: 'dot', borderWidth: 2 }},
          edges: {{ smooth: {{ type: 'curvedCW', roundness: 0.12 }} }},
          interaction: {{ hover: true, tooltipDelay: 100 }},
          physics: false
        }};
        const network = new vis.Network(container, netData, netOptions);
        network.fit({{ animation: false }});
      }}
    }}

    const weekLabels = {json.dumps(week_labels)};
    function buildLineChart(canvasId, datasets, yAxisTitle) {{
      new Chart(document.getElementById(canvasId).getContext('2d'), {{
        type: 'line',
        data: {{ labels: weekLabels, datasets: datasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: 'index', intersect: false }},
          plugins: {{
            legend: {{ position: 'bottom', labels: {{ usePointStyle: true, padding: 16 }} }},
            tooltip: {{
              callbacks: {{
                label: (ctx) => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%'
              }}
            }}
          }},
          scales: {{
            y: {{
              min: 0, max: 100,
              title: {{ display: true, text: yAxisTitle }},
              ticks: {{ callback: (v) => v + '%' }}
            }}
          }}
        }}
      }});
    }}

    buildLineChart('playoffChart', {json.dumps(playoff_datasets)}, 'Playoff Probability (%)');
    buildLineChart('lastPlaceChart', {json.dumps(last_place_datasets)}, 'Last-Place Probability (%)');

    new Chart(document.getElementById('combinedTxChart').getContext('2d'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(tx_bar_labels)},
        datasets: [
          {{ label: 'Net Trade Starter Points', data: {json.dumps(tx_trade_series)}, backgroundColor: '#16a34a', borderRadius: 4 }},
          {{ label: 'Waiver / FA Starter Points', data: {json.dumps(tx_waiver_series)}, backgroundColor: '#2563eb', borderRadius: 4 }}
        ]
      }},
      options: {{
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ position: 'bottom' }},
          tooltip: {{
            callbacks: {{
              label: (ctx) => ctx.dataset.label + ': ' + (ctx.parsed.x >= 0 ? '+' : '') + ctx.parsed.x.toFixed(1) + ' pts'
            }}
          }}
        }},
        scales: {{
          x: {{ stacked: true, title: {{ display: true, text: 'Cumulative Starter Points Added / Lost' }} }},
          y: {{ stacked: true }}
        }}
      }}
    }});

    let currentKind = 'ALL';
    function setKind(kind, btn) {{
      currentKind = kind;
      document.querySelectorAll('.sub-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyFilters();
    }}

    function applyFilters() {{
      const team = document.getElementById('teamFilter').value;
      const sort = document.getElementById('sortOrder').value;
      const container = document.getElementById('feedContainer');
      const cards = Array.from(container.getElementsByClassName('tx-card'));

      cards.forEach(c => {{
        const kindMatch = (currentKind === 'ALL' || c.dataset.kind === currentKind);
        const teamMatch = (team === 'ALL' || c.dataset.t1 === team || c.dataset.t2 === team);
        c.style.display = (kindMatch && teamMatch) ? 'block' : 'none';
      }});

      cards.sort((a, b) => {{
        if (sort === 'impact') return parseFloat(b.dataset.impact) - parseFloat(a.dataset.impact);
        return parseInt(b.dataset.ts) - parseInt(a.dataset.ts);
      }});
      cards.forEach(c => container.appendChild(c));
    }}
  </script>
</body>
</html>"""

    Path("index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    build_master_site()
