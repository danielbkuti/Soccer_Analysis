import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def process_shots(file_path: str, stat_array: list) -> tuple:
    df = pd.read_csv(file_path)

    # Position 1: Total Matches
    stat_array.append(df['game_id'].nunique())

    # Position 4 (Calculated early): Incomplete Games (Minutes < 90)
    # Note: Ensure your CSV has the 'ederson_minutes' column from the updated scraper
    incomplete_mask = df.groupby('game_id')['ederson_minutes'].transform('max') < 90
    incomplete_count = len(df.loc[incomplete_mask, 'game_id'].unique())

    # Position 2: Matches Conceded (Refined for Own Goals + Full 90 Only)
    conceded_mask = (
            (
                    ((df['team'] != 'Manchester City') & (df['result'] == 'Goal')) |
                    ((df['team'] == 'Manchester City') & (df['result'] == 'Own Goal'))
            ) &
            (df['ederson_minutes'] >= 90)
    )
    conceded_match_ids = df.loc[conceded_mask, 'game_id'].unique()
    stat_array.append(len(conceded_match_ids))

    # Build timelines for conceded matches (only for full-game goals)
    match_timelines = []
    for match_id in conceded_match_ids:
        match_data = df[df['game_id'] == match_id].copy()
        timeline = match_data[
            ((match_data['team'] != 'Manchester City') & (match_data['result'].isin(['Saved Shot', 'Goal']))) |
            ((match_data['team'] == 'Manchester City') & (match_data['result'] == 'Own Goal'))
            ].sort_values(by='minute')
        match_timelines.append(timeline)

    return match_timelines, incomplete_count


def plot_overlapping_stats(goalkeeping_records: list):
    seasons = [s[0] for s in goalkeeping_records]
    total_m = [s[1] for s in goalkeeping_records]
    conceded_m = [s[2] for s in goalkeeping_records]
    first_shot_m = [s[3] for s in goalkeeping_records]
    incomplete_m = [s[4] for s in goalkeeping_records]
    labels = ['Total Matches Played', 'Matches Conceded (Full 90)',
              'Goal on 1st Shot Faced', 'Incomplete Games (Subbed or Sent off)', 'Golden Glove Seasons' ]
    colors = ['#E0E0E0','#6CABDD', '#1C2C5B', '#11113B', '#D4A12A']

    # 1. Define Golden Glove seasons
    golden_glove_seasons = ["2019-20", "2020-21", "2021-22"]

    # 2. Assign colors: Gold if won, Skyblue otherwise
    # (Assuming Man City blue as the default)
    bar_colors = ['#D4A12A' if s in golden_glove_seasons else '#E0E0E0' for s in seasons]

    # Calculate Clean Sheets: Total - Conceded - Incomplete
    clean_sheets = [t - c - i for t, c, i in zip(total_m, conceded_m, incomplete_m)]

    x = np.arange(len(seasons))
    width = 0.6

    fig, ax = plt.subplots(figsize=(14, 8))

    # Layer 1: Total Matches
    bar1 = ax.bar(x, total_m, width, label=labels[0], color=bar_colors)

    # Layer 2: Matches Conceded
    bar2 = ax.bar(x, conceded_m, width, label=labels[1], color='#6CABDD')

    # Layer 3: Goal on 1st Shot
    bar3 = ax.bar(x, first_shot_m, width, label=labels[2], color='#1C2C5B')

    # Layer 4: Incomplete Games (Striped)
    bar4 = ax.bar(x, incomplete_m, width, label=labels[3],
                  color='#11113B', zorder=5)

    # --- FIXING ALIGNMENT AND ADDING NUMBERS ---

    # 1. Total Games label (Top of the bar)
    ax.bar_label(bar1, padding=3, weight='bold', color='#757575')

    # 2. Conceded Games label (Centered in the blue section)
    ax.bar_label(bar2, padding=3, color='black', weight='bold')

    # 3. First Shot Conceded label (Centered in the red section)
    # If the red bar is very small, we move the label slightly to avoid overlap
    ax.bar_label(bar3, padding=3, color='white', weight='bold')

    # 4. Incomplete Games label (Placed right above the striped bar)
    ax.bar_label(bar4, padding=2, color='black', weight='bold', fontsize=9)

    # 5. Clean Sheets Label (Green text at the very top)
    for i, cs in enumerate(clean_sheets):
        ax.text(i, total_m[i] + 2.5, f"CS: {cs}", ha='center', va='bottom',
                color='#2E7D32', weight='bold', fontsize=11,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Formatting
    ax.set_ylabel('Number of Matches')
    ax.set_title('Ederson Moraes Career Analysis: Efficiency & Clean Sheets')
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylim(0, max(total_m) + 8)  # Extra room for the CS labels
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    legend_elements = []
    # 4. Add a custom legend
    for i in range(len(labels)):
        legend_elements.append(Patch(facecolor=colors[i], edgecolor='black', label=labels[i]))
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('ederson_analysis_final.png')
    print("Graph generated with all values aligned.")


def check_first_conceded(game_data: list, stat_array: list, output_file=None):
    first_conceded_counter = 0
    for df in game_data:
        is_goal = df.iloc[0]['result'] == 'Goal'
        if is_goal:
            first_conceded_counter += 1

        if output_file:
            status = "GOAL on 1st shot" if is_goal else "1st shot SAVED"
            output_file.write(f"Match: {df['game'].iloc[0]} -> {status}\n")

    stat_array.append(first_conceded_counter)  # Position 3


def debug_conceded_matches(file_path: str):
    """
    Mini test function to print all matches and their status for debugging.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    games = df.groupby('game_id')

    print(f"\n{'DEBUGGING SEASON: ' + os.path.basename(file_path):^80}")
    print("-" * 90)
    print(f"{'Opponent Game':<40} | {'Mins':<5} | {'Status':<12} | {'Goals Faced'}")
    print("-" * 90)

    conceded, incomplete, clean_sheets = [], [], []

    for _, group in games:
        game_name = group['game'].iloc[0]
        max_mins = group['ederson_minutes'].max()

        # Goals conceded while on pitch (Opponent Goal or Own Goal)
        goal_mask = (
                ((group['team'] != 'Manchester City') & (group['result'] == 'Goal')) |
                ((group['team'] == 'Manchester City') & (group['result'] == 'Own Goal'))
        )
        goals_on_pitch = group[goal_mask]
        num_goals = len(goals_on_pitch)

        if max_mins < 90:
            status = "INCOMPLETE"
            incomplete.append(game_name)
        elif num_goals > 0:
            status = "CONCEDED"
            conceded.append(game_name)
        else:
            status = "CLEAN SHEET"
            clean_sheets.append(game_name)

        print(f"{game_name[:40]:<40} | {int(max_mins):<5} | {status:<12} | {num_goals}")

    print("-" * 90)
    print(
        f"Summary: {len(games)} Total | {len(conceded)} Conceded | {len(incomplete)} Incomplete | {len(clean_sheets)} Clean Sheets")
    print("-" * 90)


if __name__ == "__main__":
    career_stats = []  # Structure: [Season, Total, Conceded, 1stShot, Incomplete]
    path = r"C:\Users\danie\Projects\EdersonFootballData\Ederson Match Shots"
    save_to_txt = True

    with open("outputs/reports/Ederson_Analysis_Results.txt", "w") if save_to_txt else None as f:
        for file_name in sorted(os.listdir(path)):
            if file_name.endswith(".csv"):
                season_label = file_name[0:7]
                season_data = [season_label]

                # Process the file
                timelines, incomplete_total = process_shots(os.path.join(path, file_name), season_data)

                # Check 1st shot logic
                check_first_conceded(timelines, season_data, f)

                # Add the Incomplete count to the final position (Position 4)
                season_data.append(incomplete_total)

                career_stats.append(season_data)

                if f:
                    f.write(f"\n{season_label} Summary:\n")
                    f.write(f"- Total Games: {season_data[1]}\n")
                    f.write(f"- Full Games Conceded: {season_data[2]}\n")
                    f.write("- Games Incomplete: " + str(season_data[4]) + "\n")
                    f.write("=" * 30 + "\n")

    plot_overlapping_stats(career_stats)

    # RUN DEBUG TEST FOR 17/18
    # path_1718 = r"C:\Users\danie\Projects\EdersonFootballData\Ederson Match Shots\2017-18 Ederson Shots.csv"
    # debug_conceded_matches(path_1718)