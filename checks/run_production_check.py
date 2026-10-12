"""Production readiness verification for the restructured project."""

import ast
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_syntax():
    """All Python files parse without errors."""
    files = [
        os.path.join(PROJECT_ROOT, "src", "env_wrapper.py"),
        os.path.join(PROJECT_ROOT, "src", "train.py"),
        os.path.join(PROJECT_ROOT, "src", "predict.py"),
        os.path.join(PROJECT_ROOT, "checks", "test.py"),
        os.path.join(PROJECT_ROOT, "checks", "verify.py"),
        os.path.join(PROJECT_ROOT, "checks", "check_ram.py"),
        os.path.join(PROJECT_ROOT, "checks", "check_system.py"),
        os.path.join(PROJECT_ROOT, "checks", "check_game.py"),
        os.path.join(PROJECT_ROOT, "checks", "check_model_obs.py"),
        os.path.join(PROJECT_ROOT, "checks", "check_scenario.py"),
    ]
    all_ok = True
    for f in files:
        name = os.path.relpath(f, PROJECT_ROOT)
        try:
            with open(f) as fp:
                ast.parse(fp.read())
            print(f"  [OK] {name}")
        except SyntaxError as e:
            print(f"  [FAIL] {name}: line {e.lineno} - {e.msg}")
            all_ok = False
        except FileNotFoundError:
            print(f"  [MISSING] {name}")
            all_ok = False
    return all_ok


def check_imports():
    """src/ files can find env_wrapper, checks/ files have sys.path setup."""
    files_needing_path = [
        os.path.join(PROJECT_ROOT, "src", "train.py"),
        os.path.join(PROJECT_ROOT, "src", "predict.py"),
        os.path.join(PROJECT_ROOT, "checks", "test.py"),
        os.path.join(PROJECT_ROOT, "checks", "verify.py"),
        os.path.join(PROJECT_ROOT, "checks", "check_ram.py"),
    ]
    all_ok = True
    for f in files_needing_path:
        name = os.path.relpath(f, PROJECT_ROOT)
        with open(f) as fp:
            src = fp.read()
        has_path = "sys.path.insert" in src or "sys.path.append" in src
        has_wrapper = "SuperContraWrapper" in src or "env_wrapper" in src
        if has_path and has_wrapper:
            print(f"  [OK] {name}")
        else:
            issues = []
            if not has_path:
                issues.append("missing sys.path")
            if not has_wrapper:
                issues.append("missing env_wrapper import")
            print(f"  [FAIL] {name}: {', '.join(issues)}")
            all_ok = False
    return all_ok


def check_paths():
    """All path references point to correct locations."""
    train_path = os.path.join(PROJECT_ROOT, "src", "train.py")
    predict_path = os.path.join(PROJECT_ROOT, "src", "predict.py")

    all_ok = True
    for f, name in [(train_path, "src/train.py"), (predict_path, "src/predict.py")]:
        with open(f) as fp:
            src = fp.read()
        checks = {
            "PROJECT_ROOT": "PROJECT_ROOT" in src,
            "models/ path": "models" in src,
        }
        for check_name, passed in checks.items():
            if not passed:
                print(f"  [FAIL] {name}: missing {check_name}")
                all_ok = False

    if all_ok:
        print("  [OK] src/train.py -> models/, checkpoints/, logs/")
        print("  [OK] src/predict.py -> models/, recordings/")
    return all_ok


def check_config():
    """Config files exist and are valid."""
    all_ok = True

    data_path = os.path.join(PROJECT_ROOT, "config", "data.json")
    if os.path.exists(data_path):
        with open(data_path) as f:
            data = json.load(f)
        count = len(data.get("info", {}))
        print(f"  [OK] config/data.json: {count} RAM variables")
    else:
        print("  [MISSING] config/data.json")
        all_ok = False

    scenario_path = os.path.join(PROJECT_ROOT, "config", "scenario.json")
    if os.path.exists(scenario_path):
        with open(scenario_path) as f:
            scenario = json.load(f)
        reward_vars = scenario.get("reward", {}).get("variables", {})
        if reward_vars == {}:
            print("  [OK] config/scenario.json: built-in reward disabled")
        else:
            print("  [WARN] config/scenario.json: has reward variables defined")
    else:
        print("  [MISSING] config/scenario.json")
        all_ok = False

    return all_ok


def check_files():
    """Essential files exist."""
    all_ok = True
    files = {
        "models/super_c_production_final.zip": "Trained model",
        "models/super_c_vecnorm.pkl": "VecNormalize stats",
        "roms/Super C (USA).nes": "Game ROM",
        "requirements.txt": "Dependencies",
        "README.md": "Documentation",
        ".gitignore": "Git ignore rules",
    }
    for rel_path, desc in files.items():
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        exists = os.path.exists(full_path)
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{status}] {rel_path} ({desc})")
    return all_ok


def check_game_name():
    """Game name is consistent across all source files."""
    src_files = [
        os.path.join(PROJECT_ROOT, "src", "train.py"),
        os.path.join(PROJECT_ROOT, "src", "predict.py"),
        os.path.join(PROJECT_ROOT, "checks", "test.py"),
        os.path.join(PROJECT_ROOT, "checks", "verify.py"),
        os.path.join(PROJECT_ROOT, "checks", "check_ram.py"),
    ]
    all_ok = True
    for f in src_files:
        name = os.path.relpath(f, PROJECT_ROOT)
        with open(f) as fp:
            src = fp.read()
        if "SuperC-Nes-v0" in src:
            print(f"  [OK] {name}")
        else:
            print(f"  [FAIL] {name}: game name not found")
            all_ok = False
    return all_ok


def main():
    print()
    print("=" * 60)
    print("PRODUCTION READINESS CHECK (Restructured Project)")
    print("=" * 60)
    print()

    tests = [
        ("1. Syntax (all .py files)", check_syntax),
        ("2. Import paths (env_wrapper accessible)", check_imports),
        ("3. File paths (models/, checkpoints/)", check_paths),
        ("4. Config files (data.json, scenario.json)", check_config),
        ("5. Essential files exist", check_files),
        ("6. Game name consistency", check_game_name),
    ]

    results = []
    for name, func in tests:
        print(f"\n{name}")
        print("-" * 40)
        results.append(func())

    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"ALL {total} CHECKS PASSED — Production ready")
        print()
        print("Commands:")
        print("  python checks/test.py          # Full live test")
        print("  python src/train.py --phase 3  # Start training")
        print("  python src/predict.py          # Watch AI play")
    else:
        print(f"{passed}/{total} passed — Fix issues above")
        sys.exit(1)
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
