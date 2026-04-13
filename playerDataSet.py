"""
NBA Salary Prediction - Data Collection & EDA Starter
COSC 325 - Midterm Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

SEASON = "2026"
pd.set_option("display.max_columns", None)


def clean_bref_table(df):
    df = df[df["Player"] != "Player"].copy()
    df = df[df["Player"].notna()].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def scrape_advanced(season=SEASON):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
    print(f"Fetching advanced stats from: {url}")
    tables = pd.read_html(url)
    df = clean_bref_table(tables[0])
    keep = ["Player", "Pos", "Age", "Tm", "Team", "G", "MP",
            "PER", "TS%", "USG%", "OWS", "DWS", "WS", "WS/48",
            "OBPM", "DBPM", "BPM", "VORP"]
    df = df[[c for c in keep if c in df.columns]].copy()
    if "Team" in df.columns and "Tm" not in df.columns:
        df.rename(columns={"Team": "Tm"}, inplace=True)
    return df


def scrape_per_game(season=SEASON):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
    print(f"Fetching per-game stats from: {url}")
    tables = pd.read_html(url)
    df = clean_bref_table(tables[0])
    keep = ["Player", "Tm", "Team", "G", "GS", "MP",
            "PTS", "TRB", "AST", "STL", "BLK", "TOV", "FG%", "3P%", "FT%"]
    df = df[[c for c in keep if c in df.columns]].copy()
    if "Team" in df.columns and "Tm" not in df.columns:
        df.rename(columns={"Team": "Tm"}, inplace=True)
    return df


def scrape_salaries(season=SEASON):
    url = "https://www.basketball-reference.com/contracts/players.html"
    print(f"Fetching salaries from: {url}")
    tables = pd.read_html(url)
    df = tables[0].copy()

    if isinstance(df.columns, pd.MultiIndex):
        flattened = []
        for top, sub in df.columns:
            top = str(top).strip()
            sub = str(sub).strip()
            top = "" if top.startswith("Unnamed") else top
            sub = "" if sub.startswith("Unnamed") else sub
            flattened.append(" ".join([p for p in [top, sub] if p]).strip())
        df.columns = flattened
    else:
        df.columns = [str(c).strip() for c in df.columns]

    season_int = int(str(season))
    salary_year_label = f"{season_int - 1}-{str(season_int)[-2:]}"

    player_candidates = [c for c in df.columns if c.lower() == "player" or c.lower().endswith(" player")]
    if not player_candidates:
        raise ValueError("Could not find Player column.")
    player_col = player_candidates[0]

    salary_col = [c for c in df.columns if salary_year_label in c and "salary" in c.lower()]
    if not salary_col:
        salary_col = [c for c in df.columns if salary_year_label in c]
    if not salary_col:
        raise ValueError(f"Could not find salary column for {salary_year_label}.")
    salary_col = salary_col[0]

    df = df[[player_col, salary_col]].copy()
    df.rename(columns={player_col: "Player", salary_col: "Salary"}, inplace=True)

    df["Salary"] = (df["Salary"]
                    .astype(str)
                    .str.replace(r"[\$,]", "", regex=True)
                    .pipe(pd.to_numeric, errors="coerce"))
    df["Salary_M"] = df["Salary"] / 1_000_000

    df = df[df["Player"] != "Player"].dropna(subset=["Salary"]).copy()

    # FIX 1: contracts page has one row per team — keep highest salary only
    df = df.sort_values("Salary", ascending=False)
    df = df.drop_duplicates(subset=["Player"], keep="first")
    df.reset_index(drop=True, inplace=True)
    return df


# Scrape
df_adv = scrape_advanced()
time.sleep(4)
df_pg  = scrape_per_game()
time.sleep(4)
df_sal = scrape_salaries()


def keep_totals(df):
    # FIX 2: keep only TOT row for traded players + safety net dedup
    traded        = df[df["Tm"] == "TOT"]["Player"].unique()
    df_traded     = df[df["Player"].isin(traded) & (df["Tm"] == "TOT")]
    df_not_traded = df[~df["Player"].isin(traded)]
    result = pd.concat([df_not_traded, df_traded], ignore_index=True)
    result = result.drop_duplicates(subset=["Player"], keep="first")  # safety net
    return result

df_adv = keep_totals(df_adv)
df_pg  = keep_totals(df_pg)

print(f"Advanced unique players:  {df_adv['Player'].nunique()}")
print(f"Per-game unique players:  {df_pg['Player'].nunique()}")
print(f"Salary unique players:    {df_sal['Player'].nunique()}")

# Merge
df = df_adv.merge(df_pg.drop(columns=["Tm", "G", "MP"], errors="ignore"),
                  on="Player", how="inner")
df = df.merge(df_sal[["Player", "Salary", "Salary_M"]],
              on="Player", how="inner")

# FIX 3: final safety net after merge
df = df.drop_duplicates(subset=["Player"], keep="first")
df.reset_index(drop=True, inplace=True)

print(f"\n✅ Merged shape: {df.shape} | Unique players: {df['Player'].nunique()}")

# Convert numeric
numeric_cols = ["Age", "G", "MP", "PER", "TS%", "USG%", "OWS", "DWS",
                "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP",
                "PTS", "TRB", "AST", "STL", "BLK", "TOV",
                "FG%", "3P%", "FT%", "Salary", "Salary_M"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Salary"]).copy()
df = df[df["G"] >= 10].copy()
df.reset_index(drop=True, inplace=True)

print(f"✅ Final clean shape (>=10 games): {df.shape}")
print(df.describe())

# EDA Visualizations
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["Salary_M"], bins=40, color="royalblue", edgecolor="white")
axes[0].set_title("Salary Distribution (raw)")
axes[0].set_xlabel("Salary ($M)")
axes[0].set_ylabel("Count")
axes[1].hist(np.log1p(df["Salary_M"]), bins=40, color="darkorange", edgecolor="white")
axes[1].set_title("Salary Distribution (log-transformed)")
axes[1].set_xlabel("log(Salary + 1)")
axes[1].set_ylabel("Count")
plt.suptitle("NBA 2025-26 Salary Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("salary_distribution.png", dpi=150)
plt.show()

corr_cols = ["Salary_M", "PER", "WS", "VORP", "BPM", "USG%", "TS%", "PTS", "AST", "TRB", "Age"]
corr_cols = [c for c in corr_cols if c in df.columns]
plt.figure(figsize=(10, 7))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Heatmap — Advanced Stats vs Salary", fontsize=13)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

correlations = df[corr_cols].corr()["Salary_M"].drop("Salary_M").sort_values()
plt.figure(figsize=(8, 5))
correlations.plot(kind="barh",
                  color=["tomato" if x < 0 else "steelblue" for x in correlations])
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with Salary", fontsize=13)
plt.xlabel("Pearson Correlation")
plt.tight_layout()
plt.savefig("feature_correlation.png", dpi=150)
plt.show()

print("\n── Missing Values ──────────────────────────────")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values!")

output_file = "nba_salary_2025_26.csv"
try:
    df.to_csv(output_file, index=False)
except PermissionError:
    output_file = f"nba_salary_2025_26_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nWarning: could not overwrite nba_salary_2025_26.csv (file may be open).")

print(f"\n✅ Saved: {output_file}  |  {len(df)} players  |  {df.shape[1]} features")