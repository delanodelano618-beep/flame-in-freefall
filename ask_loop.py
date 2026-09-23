import numpy as np
import torch
from flame_in_freefall import FlamePredictor, insights, FEATURES

predictor = FlamePredictor("flame_net.pt")

print("\n=== Flame in Freefall — Ask Loop ===")
print("Type scenario values, or press Enter for defaults.")
print("Type 'quit' at any prompt to exit.\n")

while True:
    cond = {}
    defaults = {
        "oxygen_conc": 21.0,
        "pressure_kpa": 101.3,
        "fuel_droplet_mm": 1.0,
        "ignition_energy_mj": 30.0,
        "flow_velocity_cms": 1.0,
        "g_level": 1.0,
        "initial_temp_k": 295.0,
    }
    try:
        for name in FEATURES:
            raw = input(f"{name} [{defaults[name]}]: ").strip()
            if raw.lower() in ("quit", "exit", "q"):
                print("Bye.")
                raise SystemExit
            cond[name] = float(raw) if raw else defaults[name]
    except ValueError:
        print("  (not a number — using default)\n")
        continue

    p = predictor.predict(cond)
    print(f"\n-> Risk: {p['fire_risk_label']}  "
          f"(probs L/M/H/C = {[round(x,2) for x in p['fire_risk_probs']]})")
    print(f"   Spread: {p['flame_spread_rate_cms']:.3f} cm/s")
    print(f"   Soot:   {p['soot_yield']:.3f}")
    print(f"   Burn:   {p['burn_duration_s']:.1f} s")
    for line in insights(p):
        print(f"   * {line}")
    print()
