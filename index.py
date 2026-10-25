"""
NeoGuard - Fetal Heart-Trace Triage
===================================
Problem  : Birth asphyxia (a baby not getting enough oxygen during labour) is a
           leading cause of newborn death and brain injury. A cardiotocogram
           (CTG) trace can warn of it early, but people trained to read CTGs
           are scarce outside big hospitals.
Solution : A machine-learning triage assistant. A nurse or midwife types in the
           numbers from the CTG printout and gets a traffic-light result
           (Normal / Suspect / Pathological), the reasons behind it, and the
           recommended next step.
Dataset  : UCI "Cardiotocography" (2,126 CTGs labelled by 3 expert obstetricians,
           Porto, Portugal; CC BY 4.0). Downloaded automatically.

Usage
-----
    pip install pandas numpy scikit-learn ucimlrepo
    python index.py          # 1) downloads data, trains, writes model.pkl AND model.json
    python index.py serve    # 2) (optional) test the website locally on :8000

Publishing: GitHub Pages can only host static files, so the website runs the model
in the browser from model.json (an export of the pickled model). Upload index.html,
model.json (and index.py + model.pkl for the record) to GitHub and enable Pages.
"""
import io
import json
import os
import pickle
import sys
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")
JSON_PATH = os.path.join(HERE, "model.json")
THRESHOLDS = (0.25, 0.40)

FEATURES = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV", "MLTV"]
COUNT_FEATURES = ["AC", "FM", "UC", "DL", "DS", "DP"]  # stored in the dataset as events per second
CLASS_NAMES = ["normal", "suspect", "pathological"]
ARFF_URL = "https://r2.datahub.io/clu849xgz0005ic08loiq97jq/main/raw/data/cardiotocography/cardiotocography.arff"

LABELS = {
    "LB": "Baseline heart rate", "AC": "Accelerations", "FM": "Fetal movements",
    "UC": "Contractions", "DL": "Light decelerations", "DS": "Severe decelerations",
    "DP": "Prolonged decelerations", "ASTV": "Abnormal short-term variability",
    "MSTV": "Short-term variability", "ALTV": "Abnormal long-term variability",
    "MLTV": "Long-term variability",
}
RANGES = {"LB": (80, 200), "ASTV": (0, 100), "ALTV": (0, 100), "MSTV": (0, 10), "MLTV": (0, 60)}


# --------------------------------------------------------------------------- #
# 1. TRAINING
# --------------------------------------------------------------------------- #
def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names/types from either source into FEATURES + NSP."""
    df = df.copy()
    df.columns = [str(c).strip().strip("'\"") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    # The popular Kaggle copy ("fetal_health.csv") uses long column names.
    kaggle = {"baseline value": "LB", "accelerations": "AC", "fetal_movement": "FM",
              "uterine_contractions": "UC", "light_decelerations": "DL",
              "severe_decelerations": "DS", "prolongued_decelerations": "DP",
              "abnormal_short_term_variability": "ASTV",
              "mean_value_of_short_term_variability": "MSTV",
              "percentage_of_time_with_abnormal_long_term_variability": "ALTV",
              "mean_value_of_long_term_variability": "MLTV", "fetal_health": "NSP"}
    df = df.rename(columns={c: kaggle[c.lower()] for c in df.columns if c.lower() in kaggle})
    upper = {c.upper(): c for c in df.columns}
    if "NSP" not in upper or any(f not in upper for f in FEATURES):
        raise ValueError(f"expected columns not found; got {list(df.columns)[:8]}...")
    target = df[upper["NSP"]].astype(str).str.strip().replace({"N": "1", "S": "2", "P": "3"})
    out = pd.DataFrame({f: pd.to_numeric(df[upper[f]], errors="coerce") for f in FEATURES})
    out["NSP"] = pd.to_numeric(target, errors="coerce")
    out = out.dropna()
    out["NSP"] = out["NSP"].round().astype(int)
    return out


def _parse_arff(text: str) -> pd.DataFrame:
    """Tiny ARFF reader (avoids scipy's strict parser)."""
    names, rows, in_data = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@attribute"):
            names.append(s.split(None, 2)[1].strip("'\""))
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append([v.strip().strip("'\"") for v in s.split(",")])
    return pd.DataFrame(rows, columns=names)


def load_data() -> pd.DataFrame:
    # source 0: a file you downloaded yourself and put next to index.py
    for name in ("fetal_health.csv", "ctg.csv", "cardiotocography.csv", "cardiotocography.arff"):
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            try:
                print(f"Using local file: {name}")
                if name.endswith(".arff"):
                    with open(path, encoding="utf-8", errors="replace") as f:
                        return _tidy(_parse_arff(f.read()))
                return _tidy(pd.read_csv(path))
            except Exception as exc:
                print(f"  could not use {name} ({exc})")
    try:  # source 1: official UCI loader
        from ucimlrepo import fetch_ucirepo
        print("Downloading dataset from UCI (ucimlrepo)...")
        ds = fetch_ucirepo(id=193)
        return _tidy(pd.concat([ds.data.features, ds.data.targets], axis=1))
    except Exception as exc:
        print(f"  UCI loader failed ({exc}); trying mirror...")
    try:  # source 2: ARFF mirror
        req = urllib.request.Request(ARFF_URL, headers={"User-Agent": "Mozilla/5.0"})
        text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace")
        return _tidy(_parse_arff(text))
    except Exception as exc:
        print(f"  mirror failed ({exc})")
    sys.exit(
        "\nCould not download the dataset (your network may be blocking it).\n"
        "Fix: download 'fetal_health.csv' (Kaggle: 'Fetal Health Classification') or the UCI\n"
        "Cardiotocography data as CSV, save it in the same folder as index.py, and run again."
    )


def triage(p) -> int:
    """Safety-first decision rule: 0 normal, 1 suspect, 2 pathological.
    Thresholds are deliberately low so dangerous traces are rarely missed."""
    if p[2] >= THRESHOLDS[0]:
        return 2
    if p[1] + p[2] >= THRESHOLDS[1]:
        return 1
    return 0


def train() -> None:
    df = load_data()
    X, y = df[FEATURES], df["NSP"] - 1  # 0 normal, 1 suspect, 2 pathological
    print(f"Loaded {len(df)} CTG records: " +
          ", ".join(f"{n} {int((y == i).sum())}" for i, n in enumerate(CLASS_NAMES)))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=7, min_samples_leaf=3,
                                                class_weight="balanced_subsample", random_state=42),
    }
    make = lambda clf: Pipeline([("scale", StandardScaler()), ("clf", clf)])

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = {}
    for name, clf in candidates.items():
        scores[name] = cross_val_score(make(clf), X_tr, y_tr, cv=cv, scoring="f1_macro").mean()
        print(f"  {name:<20} CV macro-F1 = {scores[name]:.3f}")
    best = max(scores, key=scores.get)
    print(f"Best model: {best}")

    model = make(candidates[best]).fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    pred = np.array([triage(p) for p in proba])
    yt = y_te.to_numpy()
    metrics = {
        "accuracy": round(float(accuracy_score(yt, pred)), 3),
        "macro_f1": round(float(f1_score(yt, pred, average="macro")), 3),
        "pathological_recall": round(float((pred[yt == 2] == 2).mean()), 3),
        "abnormal_recall": round(float((pred[yt > 0] > 0).mean()), 3),
        "samples": int(len(df)),
    }
    print("Hold-out test metrics (with safety-first triage rule):", metrics)

    # Typical real example of each class for the demo buttons (30-min recording).
    examples = {}
    for k, name in enumerate(CLASS_NAMES):
        idx = np.where(yt == k)[0]
        best_i = idx[np.argmax(proba[idx, k])]
        r = X_te.iloc[best_i]
        ex = {f: (round(float(r[f] * 1800)) if f in COUNT_FEATURES else round(float(r[f]), 1))
              for f in FEATURES}
        ex["duration"] = 30
        examples[name] = ex

    final = make(candidates[best]).fit(X, y)  # refit on all data for deployment
    bundle = {
        "model": final,
        "features": FEATURES,
        "medians": X[y == 0].median().to_dict(),  # what a healthy trace looks like
        "metrics": metrics,
        "model_name": best,
        "examples": examples,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    export_json(bundle, X_te)
    print(f"\nSaved model -> {MODEL_PATH}  (pickle)")
    print(f"Saved model -> {JSON_PATH}  (used by the website)")
    print("Next: open index.html with Live Server, or run: python index.py serve")


# --------------------------------------------------------------------------- #
# 2. EXPORT: pickle -> JSON so the model can run inside the browser (GitHub Pages)
# --------------------------------------------------------------------------- #
def export_json(bundle: dict, X_check: pd.DataFrame) -> None:
    pipe = bundle["model"]
    sc, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]
    out = {
        "features": FEATURES, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "medians": {k: float(v) for k, v in bundle["medians"].items()},
        "model_name": bundle["model_name"], "examples": bundle["examples"],
        "thr": list(THRESHOLDS), **bundle["metrics"],
    }
    if isinstance(clf, LogisticRegression):
        out["type"] = "lr"
        out["coef"] = np.round(clf.coef_, 6).tolist()
        out["intercept"] = np.round(clf.intercept_, 6).tolist()
    else:
        out["type"] = "rf"
        out["trees"] = []
        for est in clf.estimators_:
            t = est.tree_
            v = t.value[:, 0, :]
            v = v / v.sum(axis=1, keepdims=True)
            out["trees"].append({
                "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                "f": t.feature.tolist(), "t": [round(float(x), 4) for x in t.threshold],
                "v": [[round(float(a), 3) for a in row] if t.children_left[i] == -1 else 0
                      for i, row in enumerate(v)],
            })
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    # Self-check: JSON model must agree with the pickled model.
    Xs = X_check.iloc[:200]
    x = (Xs.to_numpy() - np.array(out["mean"])) / np.array(out["scale"])
    ref = pipe.predict_proba(Xs)
    if out["type"] == "lr":
        z = x @ np.array(out["coef"]).T + np.array(out["intercept"])
        z = np.exp(z - z.max(axis=1, keepdims=True))
        mine = z / z.sum(axis=1, keepdims=True)
    else:
        mine = np.zeros_like(ref)
        for row_i, xi in enumerate(x):
            for t in out["trees"]:
                n = 0
                while t["f"][n] >= 0:
                    n = t["l"][n] if xi[t["f"][n]] <= t["t"][n] else t["r"][n]
                mine[row_i] += np.array(t["v"][n])
        mine /= len(out["trees"])
    print(f"Export check: largest difference between pickle and JSON = {np.abs(ref - mine).max():.4f}")


# --------------------------------------------------------------------------- #
# 3. LOCAL TEST SERVER (optional - the website itself needs no Python)
# --------------------------------------------------------------------------- #
def serve(port: int = int(os.environ.get("PORT", 8000))) -> None:
    if not os.path.exists(JSON_PATH):
        print("model.json not found - training first...")
        train()
    os.chdir(HERE)
    server = ThreadingHTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    url = f"http://localhost:{port}"
    print(f"NeoGuard running at {url}  (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve()
    else:
        train()