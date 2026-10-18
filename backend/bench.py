"""Run methods named by papers on one shared rare-event bench.

The data is synthetic and the same for every path. It is not a dataset from
the papers. Resampling is fit on training rows only. Cutoffs are chosen on a
validation slice, never on the test rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

from .methods import Path

METRIC_KEYS = (
    "average_precision",
    "f1",
    "recall",
    "precision",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
)


@dataclass
class Dataset:
    seed: int
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_rows: int
    positive_rate: float

    def as_public(self) -> dict:
        return {
            "seed": self.seed,
            "n_rows": self.n_rows,
            "positive_rate": round(self.positive_rate, 4),
            "n_fit": int(len(self.y_fit)),
            "n_val": int(len(self.y_val)),
            "n_test": int(len(self.y_test)),
            "n_pos_test": int(self.y_test.sum()),
            "n_features": int(self.x_fit.shape[1]),
            "description": (
                "Synthetic tabular bench, shared by every path. "
                "Four of eight features carry a two-cluster minority signal; the other four are noise. "
                "About 8% of rows are the rare class. "
                "This is not a dataset from the papers. It exists so the paths can be compared on the same rows."
            ),
        }


@dataclass
class StoredRun:
    path_id: str
    y_val: np.ndarray
    p_val: np.ndarray
    y_test: np.ndarray
    p_test: np.ndarray
    n_pos_before: int
    n_pos_after: int
    n_fit_rows: int
    fit_note: str


@dataclass
class Scored:
    path_id: str
    threshold: float | None
    threshold_rule: str
    feasible: bool
    metrics: dict[str, float]
    ci: dict[str, list[float]]
    fit_note: str
    n_pos_before: int
    n_pos_after: int
    n_fit_rows: int

    def as_dict(self) -> dict:
        return {
            "path_id": self.path_id,
            "threshold": None if self.threshold is None else round(float(self.threshold), 3),
            "threshold_rule": self.threshold_rule,
            "feasible": self.feasible,
            "metrics": {key: round(value, 4) for key, value in self.metrics.items()},
            "ci": {key: [round(lo, 4), round(hi, 4)] for key, (lo, hi) in self.ci.items()},
            "fit_note": self.fit_note,
            "n_pos_before": self.n_pos_before,
            "n_pos_after": self.n_pos_after,
            "n_fit_rows": self.n_fit_rows,
        }


def make_dataset(seed: int = 11) -> Dataset:
    rng = np.random.default_rng(seed)
    n_pos = 360
    n_neg = 4140
    x_neg = rng.normal(0.0, 1.0, size=(n_neg, 8))
    n1 = n_pos // 2
    n2 = n_pos - n1
    scale_a = np.array([0.85, 0.85, 0.75, 0.75, 1, 1, 1, 1])
    scale_b = np.array([0.75, 0.75, 0.7, 0.7, 1, 1, 1, 1])
    c1 = rng.normal([1.15, 1.05, 0.35, -0.15, 0, 0, 0, 0], scale_a, size=(n1, 8))
    c2 = rng.normal([-0.15, 1.25, -1.15, 0.85, 0, 0, 0, 0], scale_b, size=(n2, 8))
    x = np.vstack([x_neg, c1, c2])
    y = np.array([0] * n_neg + [1] * n_pos)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=seed, stratify=y_train)
    return Dataset(
        seed=seed,
        x_fit=x_fit,
        y_fit=y_fit,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        n_rows=int(len(y)),
        positive_rate=float(y.mean()),
    )


def experiment_source(path: Path, seed: int) -> str:
    """A self-contained script: it builds the rows and fits the model itself."""
    components = set(path.component_ids)
    imports = ["import json", "import numpy as np", "from sklearn.model_selection import train_test_split"]
    if "smote" in components:
        imports.append("from sklearn.neighbors import NearestNeighbors")
    model_id = next(
        (name for name in path.component_ids if name in {"logreg", "random_forest", "gbdt", "decision_tree"}),
        "logreg",
    )
    if "easy_ensemble" in components or model_id == "logreg":
        imports.append("from sklearn.linear_model import LogisticRegression")
    if model_id == "random_forest":
        imports.append("from sklearn.ensemble import RandomForestClassifier")
    if model_id == "gbdt":
        imports.append("from sklearn.ensemble import HistGradientBoostingClassifier")
    if model_id == "decision_tree":
        imports.append("from sklearn.tree import DecisionTreeClassifier")
    weight = "class_weight" in components
    lines = [
        '"""Synthetic rare-event table, then the model this branch fits.',
        "360 rare rows and 4140 common rows, eight numeric features, seed fixed below.",
        "This is not a dataset copied from a paper.",
        '"""',
        *imports,
        "",
        f"seed = {int(seed)}",
        "rng = np.random.default_rng(seed)",
        "n_pos, n_neg = 360, 4140",
        "x_neg = rng.normal(0.0, 1.0, size=(n_neg, 8))",
        "n1 = n_pos // 2",
        "n2 = n_pos - n1",
        "scale_a = np.array([0.85, 0.85, 0.75, 0.75, 1, 1, 1, 1])",
        "scale_b = np.array([0.75, 0.75, 0.7, 0.7, 1, 1, 1, 1])",
        "c1 = rng.normal([1.15, 1.05, 0.35, -0.15, 0, 0, 0, 0], scale_a, size=(n1, 8))",
        "c2 = rng.normal([-0.15, 1.25, -1.15, 0.85, 0, 0, 0, 0], scale_b, size=(n2, 8))",
        "x = np.vstack([x_neg, c1, c2])",
        "y = np.array([0] * n_neg + [1] * n_pos)",
        "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)",
        "x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=seed, stratify=y_train)",
        "x, y = x_fit.copy(), y_fit.copy()",
        "n_pos_before = int(y.sum())",
        "notes = []",
    ]
    if "smote" in components:
        lines.append(_SMOTE_BLOCK)
    elif "random_oversample" in components:
        lines.append(_OVERSAMPLE_BLOCK)
    elif "random_undersample" in components:
        lines.append(_UNDERSAMPLE_BLOCK)
    if "easy_ensemble" in components:
        lines.append(_EASY_BLOCK)
    else:
        lines.extend(_model_lines(model_id, weight=weight))
    lines.extend([
        "print(json.dumps({",
        "    'p_val': p_val.tolist(),",
        "    'p_test': p_test.tolist(),",
        "    'n_pos_before': n_pos_before,",
        "    'n_pos_after': int(y.sum()),",
        "    'n_fit_rows': n_fit_rows,",
        "    'fit_note': ' '.join(notes),",
        "}))",
        "",
    ])
    return "\n".join(lines)


def _model_lines(model_id: str, *, weight: bool) -> list[str]:
    weight_arg = '"balanced"' if weight else "None"
    if model_id == "logreg":
        ctor = f"model = LogisticRegression(max_iter=400, class_weight={weight_arg}, random_state=seed)"
    elif model_id == "random_forest":
        ctor = (
            "model = RandomForestClassifier(n_estimators=80, min_samples_leaf=2, "
            f"class_weight={weight_arg}, random_state=seed, n_jobs=1)"
        )
    elif model_id == "decision_tree":
        ctor = f"model = DecisionTreeClassifier(max_depth=5, class_weight={weight_arg}, random_state=seed)"
    else:
        ctor = "model = HistGradientBoostingClassifier(max_depth=4, max_iter=80, learning_rate=0.08, random_state=seed)"
    fit = ["model.fit(x, y)"]
    if weight and model_id == "gbdt":
        fit = [
            "classes, counts = np.unique(y, return_counts=True)",
            "weight = {int(c): len(y) / (len(classes) * int(cnt)) for c, cnt in zip(classes, counts)}",
            "sample_weight = np.array([weight[int(label)] for label in y])",
            "model.fit(x, y, sample_weight=sample_weight)",
            'notes.append("Gradient boosting used inverse-frequency row weights.")',
        ]
    elif weight:
        fit.append('notes.append("The classifier used inverse-frequency class weights.")')
    return [
        ctor,
        *fit,
        "classes = list(model.classes_)",
        "p_val = model.predict_proba(x_val)[:, classes.index(1)]",
        "p_test = model.predict_proba(x_test)[:, classes.index(1)]",
        "n_fit_rows = int(len(y))",
        'notes.append(f"Fit on {n_fit_rows} rows. Validation ({len(y_val)}) is held out. Test ({len(y_test)}) was scored once.")',
    ]


_SMOTE_BLOCK = """\
minority = x[y == 1]
majority_n = int((y == 0).sum())
target = int(majority_n * 0.5)
n_new = target - len(minority)
if n_new > 0 and len(minority) >= 2:
    neighbors = min(6, len(minority))
    nn = NearestNeighbors(n_neighbors=neighbors).fit(minority)
    bases = rng.integers(0, len(minority), size=n_new)
    _, indices = nn.kneighbors(minority[bases])
    synthetic = np.empty((n_new, x.shape[1]))
    for i, row in enumerate(indices):
        choices = row[row != bases[i]]
        if len(choices) == 0:
            choices = row
        partner = int(rng.choice(choices))
        gap = float(rng.random())
        synthetic[i] = minority[bases[i]] + gap * (minority[partner] - minority[bases[i]])
    x = np.vstack([x, synthetic])
    y = np.concatenate([y, np.ones(n_new, dtype=int)])
notes.append(f"SMOTE raised the minority count in the fit set from {n_pos_before} to {int(y.sum())}.")"""

_OVERSAMPLE_BLOCK = """\
pos = np.flatnonzero(y == 1)
neg_n = int((y == 0).sum())
extra = int(neg_n * 0.5) - len(pos)
if extra > 0:
    take = rng.choice(pos, size=extra, replace=True)
    x = np.vstack([x, x[take]])
    y = np.concatenate([y, np.ones(extra, dtype=int)])
notes.append(f"Random oversampling copied minority rows from {n_pos_before} to {int(y.sum())}.")"""

_UNDERSAMPLE_BLOCK = """\
pos = np.flatnonzero(y == 1)
neg = np.flatnonzero(y == 0)
keep_neg = min(len(neg), len(pos) * 3)
chosen = rng.choice(neg, size=keep_neg, replace=False)
idx = np.concatenate([pos, chosen])
rng.shuffle(idx)
x, y = x[idx], y[idx]
notes.append(f"Random undersampling kept all {n_pos_before} minority rows and cut the fit set to {len(y)} rows.")"""

_EASY_BLOCK = """\
pos = np.flatnonzero(y == 1)
neg = np.flatnonzero(y == 0)
val_scores, test_scores = [], []
for _ in range(5):
    chosen = rng.choice(neg, size=min(len(neg), max(len(pos), 1)), replace=False)
    idx = np.concatenate([pos, chosen])
    model = LogisticRegression(max_iter=400, random_state=int(rng.integers(0, 10_000)))
    model.fit(x[idx], y[idx])
    classes = list(model.classes_)
    val_scores.append(model.predict_proba(x_val)[:, classes.index(1)])
    test_scores.append(model.predict_proba(x_test)[:, classes.index(1)])
p_val = np.mean(val_scores, axis=0)
p_test = np.mean(test_scores, axis=0)
n_fit_rows = int(len(y))
notes.append("EasyEnsemble averaged five logistic regressions on undersampled majority subsets.")"""


def grounded_source(path: Path, grounding, table_path: str | None = None) -> str:
    """A script whose rows come from a paper sentence, then the branch model."""
    header = [
        '"""Rows for this branch.',
        grounding.as_public()["description"],
        "The split seed 11 is ours. The paper sentence does not name a seed.",
        '"""',
        "import json",
        "import numpy as np",
        "from sklearn.model_selection import train_test_split",
    ]
    if grounding.kind == "public" and grounding.loader == "breast_cancer":
        header.append("from sklearn.datasets import load_breast_cancer")
    elif grounding.kind == "public":
        header.append("from sklearn.datasets import fetch_openml")
    body = _ground_rows(grounding)
    components = set(path.component_ids)
    if "smote" in components:
        header.append("from sklearn.neighbors import NearestNeighbors")
    model_id = next(
        (name for name in path.component_ids if name in {"logreg", "random_forest", "gbdt", "decision_tree"}),
        "logreg",
    )
    if "easy_ensemble" in components or model_id == "logreg":
        header.append("from sklearn.linear_model import LogisticRegression")
    if model_id == "random_forest":
        header.append("from sklearn.ensemble import RandomForestClassifier")
    if model_id == "gbdt":
        header.append("from sklearn.ensemble import HistGradientBoostingClassifier")
    if model_id == "decision_tree":
        header.append("from sklearn.tree import DecisionTreeClassifier")
    lines = header + [""] + body
    if table_path:
        lines.append(_save_table(table_path))
    if "smote" in components:
        lines.append(_SMOTE_BLOCK)
    elif "random_oversample" in components:
        lines.append(_OVERSAMPLE_BLOCK)
    elif "random_undersample" in components:
        lines.append(_UNDERSAMPLE_BLOCK)
    if "easy_ensemble" in components:
        lines.append(_EASY_BLOCK)
    else:
        lines.extend(_model_lines(model_id, weight="class_weight" in components))
    lines.extend([
        "print(json.dumps({",
        "    'p_val': p_val.tolist(),",
        "    'p_test': p_test.tolist(),",
        "    'y_val': y_val.astype(int).tolist(),",
        "    'y_test': y_test.astype(int).tolist(),",
        "    'n_pos_before': n_pos_before,",
        "    'n_pos_after': int(y.sum()),",
        "    'n_fit_rows': n_fit_rows,",
        "    'fit_note': ' '.join(notes),",
        "}))",
        "",
    ])
    return "\n".join(lines)


def _save_table(table_path: str) -> str:
    return (
        "import csv\n"
        "from pathlib import Path as TablePath\n"
        f"_table = TablePath({table_path!r})\n"
        "_table.parent.mkdir(parents=True, exist_ok=True)\n"
        "with _table.open('w', newline='') as _handle:\n"
        "    _writer = csv.writer(_handle)\n"
        "    _writer.writerow([f'f{i}' for i in range(x.shape[1])] + ['label', 'split'])\n"
        "    for _name, _xs, _ys in (('fit', x, y), ('validation', x_val, y_val), ('test', x_test, y_test)):\n"
        "        for _row, _label in zip(_xs, _ys):\n"
        "            _writer.writerow([*(float(value) for value in _row), int(_label), _name])\n"
    )


def _ground_rows(grounding) -> list[str]:
    if grounding.kind == "public" and grounding.loader == "breast_cancer":
        load = [
            "bunch = load_breast_cancer()",
            "x = np.asarray(bunch.data, dtype=float)",
            "labels = np.asarray(bunch.target)",
        ]
    elif grounding.kind == "public":
        load = [
            f"bunch = fetch_openml(name={grounding.loader!r}, version=1, as_frame=False, parser='liac-arff')",
            "x = np.asarray(bunch.data, dtype=float)",
            "labels = np.asarray(bunch.target)",
        ]
    else:
        rate = grounding.positive_rate
        return [
            "seed = 11",
            "rng = np.random.default_rng(seed)",
            f"n_rows = {int(grounding.n_rows)}",
            f"n_features = {int(grounding.n_features)}",
            f"n_pos = max(1, int(round(n_rows * {rate!r})))",
            "n_neg = n_rows - n_pos",
            "x_pos = rng.normal(0.0, 1.0, size=(n_pos, n_features))",
            "x_neg = rng.normal(0.0, 1.0, size=(n_neg, n_features))",
            "x = np.vstack([x_neg, x_pos])",
            "y = np.array([0] * n_neg + [1] * n_pos)",
            'notes = ["Both classes are standard normal. The paper did not state a separation, so none was added."]',
            "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)",
            "x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=seed, stratify=y_train)",
            "x, y = x_fit.copy(), y_fit.copy()",
            "n_pos_before = int(y.sum())",
        ]
    load.extend([
        "values, counts = np.unique(labels, return_counts=True)",
        "minority = values[int(np.argmin(counts))]",
        "y = (labels == minority).astype(int)",
        "seed = 11",
        "rng = np.random.default_rng(seed)",
        "if len(values) > 2:",
        "    notes = [f'The table has {len(values)} labels. The smallest ({int(y.sum())} of {len(y)}) is the positive class.']",
        "else:",
        "    notes = [f'The smaller class ({int(y.sum())} of {len(y)} rows) is the positive label.']",
        "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)",
        "x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=seed, stratify=y_train)",
        "x, y = x_fit.copy(), y_fit.copy()",
        "n_pos_before = int(y.sum())",
    ])
    return load


def run_experiment_code(path: Path, data: Dataset | None = None, grounding=None, table_path: str | None = None) -> tuple[StoredRun, str, str]:
    """Write the branch to experiments/ and run that file. Probabilities come from the process."""
    import os
    import subprocess
    import sys
    from pathlib import Path as FilePath

    if grounding is not None:
        source = grounded_source(path, grounding, table_path)
    else:
        if data is None:
            raise RuntimeError("A branch needs either a paper's data or the comparison table.")
        source = experiment_source(path, data.seed)
    folder = FilePath(__file__).resolve().parents[1] / "experiments"
    folder.mkdir(exist_ok=True)
    script = folder / f"{path.path_id}.py"
    script.write_text(source)
    import certifi

    env = os.environ.copy()
    env["PYTHONPATH"] = str(folder.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["SSL_CERT_FILE"] = certifi.where()
    env["REQUESTS_CA_BUNDLE"] = certifi.where()
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(folder.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no output").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"exit {completed.returncode}")
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    if "y_val" in payload:
        y_val = np.asarray(payload["y_val"])
        y_test = np.asarray(payload["y_test"])
    else:
        y_val = data.y_val
        y_test = data.y_test
    stored = StoredRun(
        path_id=path.path_id,
        y_val=y_val,
        p_val=np.asarray(payload["p_val"], dtype=float),
        y_test=y_test,
        p_test=np.asarray(payload["p_test"], dtype=float),
        n_pos_before=int(payload["n_pos_before"]),
        n_pos_after=int(payload["n_pos_after"]),
        n_fit_rows=int(payload["n_fit_rows"]),
        fit_note=str(payload["fit_note"]),
    )
    if len(stored.p_test) != len(stored.y_test) or len(stored.p_val) != len(stored.y_val):
        raise RuntimeError("The script returned a different number of probabilities than labels.")
    return stored, source, str(script.relative_to(folder.parent))


def run_path(path: Path, data: Dataset) -> StoredRun:
    rng = np.random.default_rng(data.seed)
    x, y = data.x_fit.copy(), data.y_fit.copy()
    n_pos_before = int(y.sum())
    notes: list[str] = []
    components = set(path.component_ids)

    if "smote" in components:
        x, y = smote(x, y, rng)
        notes.append(f"SMOTE raised the minority count in the fit set from {n_pos_before} to {int(y.sum())}.")
    elif "random_oversample" in components:
        x, y = random_oversample(x, y, rng)
        notes.append(f"Random oversampling copied minority rows from {n_pos_before} to {int(y.sum())}.")
    elif "random_undersample" in components:
        x, y = random_undersample(x, y, rng)
        notes.append(
            f"Random undersampling kept all {n_pos_before} minority rows and cut the fit set to {len(y)} rows."
        )

    if "easy_ensemble" in components:
        p_val, p_test = easy_ensemble(x, y, data, rng)
        notes.append("EasyEnsemble averaged five logistic regressions on undersampled majority subsets.")
    else:
        model_id = next(component for component in path.component_ids if component in {
            "logreg", "random_forest", "gbdt", "decision_tree"
        })
        weight = "class_weight" in components
        model = _model(model_id, weight=weight, seed=data.seed)
        if weight and model_id == "gbdt":
            model.fit(x, y, sample_weight=_balanced_weights(y))
            notes.append("Gradient boosting used inverse-frequency row weights.")
        else:
            model.fit(x, y)
            if weight:
                notes.append("The classifier used inverse-frequency class weights.")
        p_val = _positive_proba(model, data.x_val)
        p_test = _positive_proba(model, data.x_test)
        notes.append(f"Fit on {len(y)} rows. Validation ({len(data.y_val)}) picked cutoffs. Test ({len(data.y_test)}) was scored once.")

    return StoredRun(
        path_id=path.path_id,
        y_val=data.y_val,
        p_val=p_val,
        y_test=data.y_test,
        p_test=p_test,
        n_pos_before=n_pos_before,
        n_pos_after=int(y.sum()),
        n_fit_rows=int(len(y)),
        fit_note=" ".join(notes),
    )


def score_run(
    stored: StoredRun,
    *,
    components: list[str],
    precision_floor: float | None,
    bootstrap_seed: int = 11,
) -> Scored:
    """Turn stored probabilities into the operating point a person asked for."""
    tunes_threshold = "threshold" in components or precision_floor is not None
    feasible = True
    if precision_floor is not None:
        threshold, feasible = _threshold_for_floor(stored.y_val, stored.p_val, precision_floor)
        rule = (
            f"Cutoff {threshold:.2f} maximized validation recall among cutoffs with precision ≥ {precision_floor:.2f}."
            if feasible
            else f"No validation cutoff held precision ≥ {precision_floor:.2f}. Showing the most precise cutoff instead."
        )
    elif "threshold" in components:
        threshold = _threshold_max_f1(stored.y_val, stored.p_val)
        rule = f"Cutoff {threshold:.2f} maximized F1 on the validation slice."
    else:
        threshold = 0.5
        rule = "Cutoff 0.5, the classifier default. This path does not tune a threshold."

    pred = (stored.p_test >= threshold).astype(int)
    metrics = _metrics(stored.y_test, pred, stored.p_test)
    if precision_floor is not None and metrics["precision"] + 1e-9 < precision_floor:
        rule += (
            f" On the test rows precision is {metrics['precision']:.2f}, below {precision_floor:.2f}. "
            "The cutoff stays where the validation slice put it."
        )
    ci = {
        "average_precision": bootstrap_ci(stored.y_test, stored.p_test, "average_precision", seed=bootstrap_seed),
    }
    if tunes_threshold or True:
        ci["recall"] = bootstrap_ci_at_threshold(stored.y_test, stored.p_test, threshold, "recall", seed=bootstrap_seed)
    return Scored(
        path_id=stored.path_id,
        threshold=float(threshold),
        threshold_rule=rule,
        feasible=feasible,
        metrics=metrics,
        ci=ci,
        fit_note=stored.fit_note,
        n_pos_before=stored.n_pos_before,
        n_pos_after=stored.n_pos_after,
        n_fit_rows=stored.n_fit_rows,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, metric: str, *, seed: int, n: int = 400) -> list[float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    size = len(y)
    for _ in range(n):
        idx = rng.integers(0, size, size)
        if y[idx].sum() == 0 or y[idx].sum() == size:
            continue
        if metric == "average_precision":
            values.append(float(average_precision_score(y[idx], scores[idx])))
    if len(values) < 20:
        point = float(average_precision_score(y, scores))
        return [point, point]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_ci_at_threshold(
    y: np.ndarray, scores: np.ndarray, threshold: float, metric: str, *, seed: int, n: int = 400
) -> list[float]:
    rng = np.random.default_rng(seed + 1)
    values: list[float] = []
    size = len(y)
    for _ in range(n):
        idx = rng.integers(0, size, size)
        if y[idx].sum() == 0:
            continue
        pred = (scores[idx] >= threshold).astype(int)
        if metric == "recall":
            values.append(float(recall_score(y[idx], pred, zero_division=0)))
        elif metric == "f1":
            values.append(float(f1_score(y[idx], pred, zero_division=0)))
    if len(values) < 20:
        return [0.0, 0.0]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def _threshold_max_f1(y: np.ndarray, scores: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for threshold in _grid():
        pred = (scores >= threshold).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(threshold)
    return best_t


def _threshold_for_floor(y: np.ndarray, scores: np.ndarray, floor: float) -> tuple[float, bool]:
    best: tuple[float, float] | None = None
    fallback: tuple[float, float] | None = None
    for threshold in _grid():
        pred = (scores >= threshold).astype(int)
        precision = float(precision_score(y, pred, zero_division=0))
        recall = float(recall_score(y, pred, zero_division=0))
        if fallback is None or precision > fallback[0]:
            fallback = (precision, float(threshold))
        if precision + 1e-9 >= floor and (best is None or recall > best[0]):
            best = (recall, float(threshold))
    if best:
        return best[1], True
    return (fallback[1] if fallback else 0.5), False


def _grid() -> list[float]:
    return [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]


def _model(model_id: str, *, weight: bool, seed: int):
    class_weight = "balanced" if weight else None
    if model_id == "logreg":
        return LogisticRegression(max_iter=400, class_weight=class_weight, random_state=seed)
    if model_id == "random_forest":
        return RandomForestClassifier(
            n_estimators=80,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=1,
        )
    if model_id == "decision_tree":
        return DecisionTreeClassifier(max_depth=5, class_weight=class_weight, random_state=seed)
    if model_id == "gbdt":
        return HistGradientBoostingClassifier(max_depth=4, max_iter=80, learning_rate=0.08, random_state=seed)
    raise KeyError(model_id)


def _positive_proba(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    # The positive class is the column labeled 1.
    classes = list(model.classes_)
    return proba[:, classes.index(1)]


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    weight = {int(c): n / (len(classes) * int(cnt)) for c, cnt in zip(classes, counts)}
    return np.array([weight[int(label)] for label in y])


def smote(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, k: int = 5, ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    minority = x[y == 1]
    majority_n = int((y == 0).sum())
    if len(minority) < 2:
        return x, y
    target = int(majority_n * ratio)
    n_new = target - len(minority)
    if n_new <= 0:
        return x, y
    neighbors = min(k + 1, len(minority))
    nn = NearestNeighbors(n_neighbors=neighbors).fit(minority)
    bases = rng.integers(0, len(minority), size=n_new)
    _, indices = nn.kneighbors(minority[bases])
    synthetic = np.empty((n_new, x.shape[1]))
    for i, row in enumerate(indices):
        choices = row[row != bases[i]]
        if len(choices) == 0:
            choices = row
        partner = int(rng.choice(choices))
        gap = float(rng.random())
        synthetic[i] = minority[bases[i]] + gap * (minority[partner] - minority[bases[i]])
    y_new = np.ones(n_new, dtype=int)
    return np.vstack([x, synthetic]), np.concatenate([y, y_new])


def random_oversample(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero(y == 1)
    neg_n = int((y == 0).sum())
    extra = int(neg_n * ratio) - len(pos)
    if extra <= 0:
        return x, y
    take = rng.choice(pos, size=extra, replace=True)
    return np.vstack([x, x[take]]), np.concatenate([y, np.ones(extra, dtype=int)])


def random_undersample(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep_neg = min(len(neg), len(pos) * 3)
    chosen = rng.choice(neg, size=keep_neg, replace=False)
    idx = np.concatenate([pos, chosen])
    rng.shuffle(idx)
    return x[idx], y[idx]


def easy_ensemble(x: np.ndarray, y: np.ndarray, data: Dataset, rng: np.random.Generator, bags: int = 5) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    val_scores = []
    test_scores = []
    for _ in range(bags):
        chosen = rng.choice(neg, size=min(len(neg), max(len(pos), 1)), replace=False)
        idx = np.concatenate([pos, chosen])
        model = LogisticRegression(max_iter=400, random_state=int(rng.integers(0, 10_000)))
        model.fit(x[idx], y[idx])
        val_scores.append(_positive_proba(model, data.x_val))
        test_scores.append(_positive_proba(model, data.x_test))
    return np.mean(val_scores, axis=0), np.mean(test_scores, axis=0)

