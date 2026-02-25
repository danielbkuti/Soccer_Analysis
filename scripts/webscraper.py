#import soccerdata as sd
import pandas as pd
#import time
# import os


def get_season_match_ids(season):
    ederson_ID = 6054
    season_shot_data = []
    season_upd =  season.split("/")[0][2:] + season.split("/")[1]
    underStat = sd.Understat(leagues="ENG-Premier League", seasons=season_upd)
    print(f"\n\nBeginning processing data for the {season} season ({season_upd})\n\n")
    season_schedule = underStat.read_schedule()
    condition = (season_schedule['home_team'] == 'Manchester City') | (season_schedule['away_team'] == 'Manchester City')

    # 2. Filter the DataFrame and select the 'game_id' column
    man_city_game_ids = season_schedule[condition]['game_id']

    game_id_list = man_city_game_ids.tolist()
    if len(game_id_list) == 38:
        print("We found all the games for the season!")
    else:
        raise ValueError("We did not find all the games for the season!")

    for game in game_id_list:
        time.sleep(4)
        checker = underStat.read_player_match_stats(game)
        game_name = checker.reset_index()
        game_name = game_name['game'][0]
        print(f"Processing data for {game_name}...")
        if ederson_ID in checker['player_id'].values:
            # --- NEW CODE TO TRACK MINUTES ---
            ederson_stats = checker[checker['player_id'] == ederson_ID]
            minutes_played = ederson_stats['minutes'].iloc[0]

            match_shot_data = underStat.read_shot_events(game)
            # Add a new column to the shot data for this game
            match_shot_data['ederson_minutes'] = minutes_played
            # ---------------------------------

            season_shot_data.append(match_shot_data)
        else:
            print(f"Ederson did not play {game_name}")

    final_df = pd.concat(season_shot_data)
    return final_df


def generate_season_csv(season, shots_df):
    # 1. Define your target folder
    folder_path = r"C:\Users\danie\Projects\EdersonFootballData\Ederson Match Shots"

    # 2. Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory: {folder_path}")

    # 3. Format the filename
    safe_season = season.replace("/", "-")
    file_name = f"{safe_season} Ederson Shots.csv"

    # 4. Join the folder path and filename
    full_output_path = os.path.join(folder_path, file_name)

    # 5. Save the CSV
    shots_df.to_csv(full_output_path)
    print(f"Success! Saved to: {full_output_path}")
    #time.sleep(60)


if __name__ == "__main__":
    season_arr = []
    for i in range(17,25):
        season = "20"+str(i)+"/"+str(i+1)

    for season in season_arr:

        shots_df = get_season_match_ids(season)

        if shots_df is not None:
            generate_season_csv(season, shots_df)

    # season = "2021/22"
    # shots_df = get_season_match_ids(season)
    # generate_season_csv(season, shots_df)

    # df = pd.read_csv(
    #     "../data/raw/2017-18_ederson_shots.csv",
    #     encoding="cp1252",  # handles Windows files
    #     engine="python",
    #     encoding_errors="replace"
    # )
    #
    # df.to_csv(
    #     "../data/raw/2017-18_ederson_shots_utf8.csv",
    #     index=False,
    #     encoding="utf-8"
    # )
    #
    # print("UTF-8 file written")