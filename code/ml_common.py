"""leakage-safe, subject-level repeated cross-validation utilities.

Input tables are ONE ROW PER SUBJECT, so a stratified split is automatically a
subject-level split. We pass RID as `groups`, assert uniqueness, and assert no
train/test subject overlap per fold. Preprocessing lives inside the pipeline
(fit on train folds only). Hyper-parameters fixed a priori for speed/fairness.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import (RepeatedStratifiedKFold, GridSearchCV,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.metrics import (balanced_accuracy_score, f1_score, accuracy_score,
                             roc_auc_score, confusion_matrix, roc_curve)


def make_models():
    """name -> (estimator, param_grid_or_None, needs_scaling)"""
    return {
        "logistic_l2": (
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=3000,
                              class_weight="balanced"))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]}, True),
        "extra_trees": (
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("clf", ExtraTreesClassifier(n_estimators=300, n_jobs=2,
                              class_weight="balanced_subsample", random_state=0))]),
            None, False),
        "hist_gb": (
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("clf", HistGradientBoostingClassifier(learning_rate=0.15,
                              max_iter=25, l2_regularization=1.0,
                              early_stopping=False, random_state=0))]),
            None, False),
        "random_forest": (
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("clf", RandomForestClassifier(n_estimators=200, n_jobs=2,
                              class_weight="balanced_subsample", random_state=0))]),
            None, False),
        "knn": (
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("clf", KNeighborsClassifier(n_neighbors=15))]),
            None, True),
    }


def fold_metrics(y_true, y_pred, proba, classes):
    from sklearn.preprocessing import label_binarize
    import numpy as np
    n = len(classes)
    out = dict(balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
               macro_f1=f1_score(y_true, y_pred, average="macro"),
               accuracy=accuracy_score(y_true, y_pred))
    try:
        if n == 2:
            ybin = (np.asarray(y_true) == classes[1]).astype(int)
            out["roc_auc"] = roc_auc_score(ybin, proba[:, 1])
        else:
            Y = label_binarize(y_true, classes=classes)
            out["roc_auc_ovr_macro"] = roc_auc_score(Y, proba, average="macro")
            out["roc_auc_ovr_weighted"] = roc_auc_score(Y, proba, average="weighted")
    except Exception:
        pass
    return out


def evaluate_combo(X, y, groups, classes, model_name,
                   n_splits=5, n_repeats=2, seed=42, tune=False):
    """Repeated subject-level CV for one model. Returns (folds_df, oof_df)."""
    assert pd.Series(groups).is_unique, "groups (RID) must be unique = 1 row/subject"
    est, grid, _ = make_models()[model_name]
    yv = pd.Series(y).astype(str).values
    Xv = X.values if hasattr(X, "values") else X
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=seed)
    folds, oof = [], []
    g = np.asarray(groups)
    for i, (tr, te) in enumerate(rkf.split(Xv, yv)):
        assert not (set(g[tr]) & set(g[te])), "subject overlap train/test!"
        rep = i // n_splits
        if tune and grid is not None:
            gs = GridSearchCV(est, grid, scoring="balanced_accuracy",
                              cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                              n_jobs=-1)
            gs.fit(Xv[tr], yv[tr])
            mdl = gs.best_estimator_
        else:
            mdl = est.fit(Xv[tr], yv[tr])
        proba = mdl.predict_proba(Xv[te])
        cl = list(mdl.classes_)
        proba = proba[:, [cl.index(c) for c in classes]]
        pred = np.array(classes)[proba.argmax(1)]
        m = fold_metrics(yv[te], pred, proba, classes)
        m.update(dict(model=model_name, fold=i, repeat=rep, n_test=len(te)))
        folds.append(m)
        if rep == 0:
            for j, idx in enumerate(te):
                row = dict(model=model_name, RID=g[idx], y_true=yv[idx], y_pred=pred[j])
                for k, c in enumerate(classes):
                    row["p_" + c] = proba[j, k]
                oof.append(row)
    return pd.DataFrame(folds), pd.DataFrame(oof)


def opspoint(y, p, pos_label=1):
    """Youden-J operating point from cross-validated OOF probabilities."""
    fpr, tpr, thr = roc_curve(y, p)
    t = thr[int(np.argmax(tpr - fpr))]
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return dict(threshold=float(t), sensitivity=sens, specificity=spec,
                ppv=ppv, npv=npv)


def summarize(folds_df, classes=None):
    excl = ("task", "featureset", "model", "fold", "repeat", "n_test")
    metric_cols = [c for c in folds_df.columns if c not in excl]
    rows = []
    for model, g in folds_df.groupby("model"):
        r = {"model": model, "n_folds": len(g)}
        for mc in metric_cols:
            v = pd.to_numeric(g[mc], errors="coerce").dropna().values
            if len(v) == 0:
                continue
            mean, sd = v.mean(), v.std(ddof=1)
            half = 1.96 * sd / np.sqrt(len(v))
            r[mc + "_mean"] = mean
            r[mc + "_sd"] = sd
            r[mc + "_lo"] = mean - half
            r[mc + "_hi"] = mean + half
        rows.append(r)
    return pd.DataFrame(rows)
