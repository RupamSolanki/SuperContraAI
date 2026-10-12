"""Check which games are registered in stable-retro."""

import stable_retro

games = stable_retro.data.list_games()
matches = [g for g in games if "super" in g.lower() or "contra" in g.lower()]
print(f"Total games: {len(games)}")
print(f"Contra/Super matches: {matches}")
