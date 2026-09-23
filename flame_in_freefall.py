import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

FEATURES = ["oxygen_conc","pressure_kpa","fuel_droplet_mm","ignition_energy_mj",
            "flow_velocity_cms","g_level","initial_temp_k"]
TARGETS_REG = ["flame_spread_rate","burn_duration_s","soot_yield"]
TARGET_CLS = "fire_risk_class"
RISK_LABELS = {0:"LOW",1:"MODERATE",2:"HIGH",3:"CRITICAL"}

def generate_dataset(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "oxygen_conc": rng.uniform(12,30,n_samples),
        "pressure_kpa": rng.uniform(30,150,n_samples),
        "fuel_droplet_mm": rng.uniform(0.5,4.0,n_samples),
        "ignition_energy_mj": rng.uniform(5,200,n_samples),
        "flow_velocity_cms": rng.uniform(0.0,10.0,n_samples),
        "g_level": rng.uniform(0.0,50.0,n_samples),
        "initial_temp_k": rng.uniform(280,340,n_samples),
    })
    df["flame_spread_rate"] = np.clip(0.05 + 0.012*(df.oxygen_conc-15)
        + 0.0008*(df.pressure_kpa-50) + 0.09*df.flow_velocity_cms
        + 0.03*df.fuel_droplet_mm - 0.0004*df.g_level
        + rng.normal(0,0.02,n_samples), 0.0, 3.0)
    df["burn_duration_s"] = np.clip(12.0 + 45.0*df.fuel_droplet_mm
        - 0.9*(df.oxygen_conc-21) - 0.05*(df.pressure_kpa-100)
        + 8.0*np.exp(-df.flow_velocity_cms) + rng.normal(0,3.0,n_samples), 1.0, 200.0)
    df["soot_yield"] = np.clip(0.4 - 0.02*(df.oxygen_conc-21)
        + 0.003*(df.pressure_kpa-100) + 0.05*df.fuel_droplet_mm
        + rng.normal(0,0.03,n_samples), 0.0, 1.0)
    score = (1.8*df.flame_spread_rate + 2.2*df.soot_yield
             + 0.06*(df.oxygen_conc-15) + 0.5*df.flow_velocity_cms)
    df["fire_risk_class"] = np.digitize(score, bins=[1.5,2.5,3.6]).astype(np.int64)
    return df

class FlameNet(nn.Module):
    def __init__(self, n_features, n_classes=4, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2), nn.SiLU(),
        )
        self.head_spread = nn.Linear(hidden//2, 1)
        self.head_dur = nn.Linear(hidden//2, 1)
        self.head_soot = nn.Linear(hidden//2, 1)
        self.head_risk = nn.Linear(hidden//2, n_classes)
    def forward(self, x):
        h = self.trunk(x)
        return {"spread": self.head_spread(h).squeeze(-1),
                "dur": self.head_dur(h).squeeze(-1),
                "soot": self.head_soot(h).squeeze(-1),
                "risk": self.head_risk(h)}

def train(epochs=40, batch_size=128, lr=1e-3, ckpt="flame_net.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = generate_dataset()
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/microgravity_combustion.csv", index=False)
    X = df[FEATURES].values.astype(np.float32)
    y_reg = df[TARGETS_REG].values.astype(np.float32)
    y_cls = df[TARGET_CLS].values.astype(np.int64)
    reg_scaler = StandardScaler().fit(y_reg)
    y_reg_n = reg_scaler.transform(y_reg).astype(np.float32)
    X_tr, X_va, yr_tr, yr_va, yc_tr, yc_va = train_test_split(
        X, y_reg_n, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    feat_scaler = StandardScaler().fit(X_tr)
    X_tr = feat_scaler.transform(X_tr).astype(np.float32)
    X_va = feat_scaler.transform(X_va).astype(np.float32)
    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr),
        torch.from_numpy(yr_tr), torch.from_numpy(yc_tr)),
        batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X_va),
        torch.from_numpy(yr_va), torch.from_numpy(yc_va)), batch_size=batch_size)
    model = FlameNet(len(FEATURES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    best_val = float("inf")
    for epoch in range(1, epochs+1):
        model.train(); tr = 0.0
        for xb, yrb, ycb in train_dl:
            xb, yrb, ycb = xb.to(device), yrb.to(device), ycb.to(device)
            opt.zero_grad()
            o = model(xb)
            loss = (mse(o["spread"], yrb[:,0]) + mse(o["dur"], yrb[:,1])
                    + mse(o["soot"], yrb[:,2]) + 0.5*ce(o["risk"], ycb))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tr += loss.item()*xb.size(0)
        sched.step()
        model.eval(); vl, corr, tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yrb, ycb in val_dl:
                xb, yrb, ycb = xb.to(device), yrb.to(device), ycb.to(device)
                o = model(xb)
                vl += (mse(o["spread"], yrb[:,0]) + mse(o["dur"], yrb[:,1])
                       + mse(o["soot"], yrb[:,2]) + 0.5*ce(o["risk"], ycb)).item()*xb.size(0)
                corr += (o["risk"].argmax(1) == ycb).sum().item()
                tot += xb.size(0)
        vl /= len(val_dl)
        print(f"Epoch {epoch:3d} | train {tr/len(train_dl):.4f} | val {vl:.4f} | acc {corr/tot:.3f}")
        if vl < best_val:
            best_val = vl
            torch.save({"model_state": model.state_dict(),
                        "feat_mean": feat_scaler.mean_, "feat_scale": feat_scaler.scale_,
                        "reg_mean": reg_scaler.mean_, "reg_scale": reg_scaler.scale_}, ckpt)
    print(f"Best val loss {best_val:.4f} -> saved {ckpt}")
    return ckpt

class FlamePredictor:
    def __init__(self, ckpt="flame_net.pt"):
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = FlameNet(len(FEATURES))
        self.model.load_state_dict(c["model_state"]); self.model.eval()
        self.fm, self.fs = c["feat_mean"], c["feat_scale"]
        self.rm, self.rs = c["reg_mean"], c["reg_scale"]
    def predict(self, cond):
        x = np.array([[cond[f] for f in FEATURES]], dtype=np.float32)
        x = ((x - self.fm) / self.fs).astype(np.float32)
        with torch.no_grad():
            o = self.model(torch.from_numpy(x))
        reg = np.stack([o["spread"].item(), o["dur"].item(), o["soot"].item()])
        reg = reg*self.rs + self.rm
        probs = torch.softmax(o["risk"], -1).numpy()[0]
        cls = int(probs.argmax())
        return {"flame_spread_rate_cms": float(reg[0]),
                "burn_duration_s": float(reg[1]),
                "soot_yield": float(reg[2]),
                "fire_risk_class": cls,
                "fire_risk_label": RISK_LABELS[cls],
                "fire_risk_probs": probs.tolist()}

def insights(pred):
    out, r = [], pred["fire_risk_class"]
    s, so, d = pred["flame_spread_rate_cms"], pred["soot_yield"], pred["burn_duration_s"]
    if r >= 3: out.append("CRITICAL: Suppress immediately. Microgravity fires spread spherically.")
    elif r == 2: out.append("HIGH RISK: Isolate module, reduce O2, prep suppression.")
    if s > 1.0: out.append(f"Fast spread ({s:.2f} cm/s) - cut cabin fans near source.")
    if so > 0.5: out.append(f"High soot ({so:.2f}) - engage HEPA filtration.")
    if d > 60: out.append(f"Long burn estimate ({d:.0f}s) - quench early.")
    if not out: out.append("Conditions within safe margins.")
    out.append("Reference: NASA CFM & FLEX microgravity combustion studies.")
    return out

def ask():
    predictor = FlamePredictor("flame_net.pt")
    print("\n=== Type your scenario values (Enter = use default) ===")
    cond = {
        "oxygen_conc": float(input("O2 %          [21]: ") or 21),
        "pressure_kpa": float(input("Pressure kPa  [101.3]: ") or 101.3),
        "fuel_droplet_mm": float(input("Droplet mm    [1.0]: ") or 1.0),
        "ignition_energy_mj": float(input("Spark mJ      [30]: ") or 30),
        "flow_velocity_cms": float(input("Airflow cm/s  [1.0]: ") or 1.0),
        "g_level": float(input("g level (ug)  [1.0]: ") or 1.0),
        "initial_temp_k": float(input("Temp K        [295]: ") or 295),
    }
    p = predictor.predict(cond)
    print(f"\n-> Risk: {p['fire_risk_label']}")
    print(f"   Spread: {p['flame_spread_rate_cms']:.3f} cm/s")
    print(f"   Soot:   {p['soot_yield']:.3f}")
    print(f"   Burn:   {p['burn_duration_s']:.1f} s")
    for line in insights(p): print(f"   * {line}")

if __name__ == "__main__":
    train(epochs=40)
    ask()
