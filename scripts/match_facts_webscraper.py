import soccerdata as sd

fbref = sd.FBref(
    leagues="ENG-Premier League",
    seasons=["2022-23", "2023-24", "2024-25"],
    no_cache=True
)
schedule = fbref.read_schedule()
schedule.head()