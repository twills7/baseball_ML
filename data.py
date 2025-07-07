import requests
import sqlite3

def get_all_players():
    url = "https://api.sleeper.app/v1/players/nfl"
    try:
        response = requests.get(url)
        response.raise_for_status()
        players = response.json()
        return players
    except Exception as e:
        print(f"Error fetching players: {e}")
        return {}
    
def get_nfl_matchups():
    url = "https://api.sleeper.app/v1/league/NFL/matchups"
    try:
        response = requests.get(url)
        response.raise_for_status()
        matchups = response.json()
        return matchups
    except Exception as e:
        print(f"Error fetching matchups: {e}")
        return []
    
if __name__ == "__main__":
    players = get_all_players()
    if players:
        print(f"Fetched {len(players)} players from Sleeper API.")
    else:
        print("No players found or error occurred.")

    matchups = get_nfl_matchups()
    if matchups:
        print(f"Fetched {len(matchups)} matchups from Sleeper API.")
    else:
        print("No matchups found or error occurred.")
    
    # Create a SQLite database or connect to one
    conn = sqlite3.connect('nfl_players.db')
    cursor = conn.cursor()

    # Create a table for players
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS players (
        player_id TEXT PRIMARY KEY,
        full_name TEXT,
        first_name TEXT,
        last_name TEXT,
        position TEXT,
        team TEXT,
        age INTEGER,
        height TEXT,
        weight TEXT,
        college TEXT,
        status TEXT,
        hashtag TEXT,
        depth_chart_position INTEGER,
        sport TEXT,
        fantasy_positions TEXT,
        number INTEGER,
        search_last_name TEXT,
        injury_start_date TEXT,
        practice_participation TEXT,
        sportradar_id TEXT,
        fantasy_data_id INTEGER,
        injury_status TEXT,
        stats_id TEXT,
        birth_country TEXT,
        espn_id TEXT,
        search_rank INTEGER,
        depth_chart_order INTEGER,
        years_exp INTEGER,
        rotowire_id TEXT,
        rotoworld_id INTEGER,
        search_first_name TEXT,
        yahoo_id TEXT
    )
    ''')

    # Insert players into the database
    for player_id, player_data in players.items():
        cursor.execute('''
        INSERT OR REPLACE INTO players (
            player_id, full_name, first_name, last_name, position, team, age, height, weight, college, status,
            hashtag, depth_chart_position, sport, fantasy_positions, number, search_last_name, injury_start_date,
            practice_participation, sportradar_id, fantasy_data_id, injury_status, stats_id, birth_country, espn_id,
            search_rank, depth_chart_order, years_exp, rotowire_id, rotoworld_id, search_first_name, yahoo_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            player_id,
            player_data.get('search_full_name'),
            player_data.get('search_first_name'),
            player_data.get('search_last_name'),
            player_data.get('position'),
            player_data.get('team'),
            player_data.get('age'),
            player_data.get('height'),
            player_data.get('weight'),
            player_data.get('college'),
            player_data.get('status'),
            player_data.get('hashtag'),
            player_data.get('depth_chart_position'),
            player_data.get('sport'),
            f"{player_data.get('fantasy_positions', [])}",
            player_data.get('number'),
            player_data.get('search_last_name'),
            player_data.get('injury_start_date'),
            player_data.get('practice_participation'),
            player_data.get('sportradar_id'),
            player_data.get('fantasy_data_id'),
            player_data.get('injury_status'),
            player_data.get('stats_id'),
            player_data.get('birth_country'),
            player_data.get('espn_id'),
            player_data.get('search_rank'),
            player_data.get('depth_chart_order'),
            player_data.get('years_exp'),
            player_data.get('rotowire_id'),
            player_data.get('rotoworld_id'),
            player_data.get('search_first_name'),
            player_data.get('yahoo_id')
        ))

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    print("Players have been added to the database.")