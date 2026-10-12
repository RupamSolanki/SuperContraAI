"""Check what scenario/data files exist for SuperC-Nes-v0."""
import stable_retro
import os

game_path = os.path.join(os.path.dirname(stable_retro.__file__), 'data', 'stable', 'SuperC-Nes-v0')
print(f"Game path: {game_path}")
print(f"Files:")
for f in os.listdir(game_path):
    print(f"  {f}")
    if f.endswith('.json'):
        with open(os.path.join(game_path, f)) as fp:
            print(f"    Content: {fp.read()[:500]}")
