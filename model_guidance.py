"""Local, rule-based modeling advice, never training or a model ranking.

Public API: advise_models(profile, goal="Explore", target=None) -> dict.
All list fields contain strings. ``status`` is one of: exploration, advisory,
needs_target, invalid_target, needs_review, insufficient_data, unsupported_goal.
"advisory" means conditional suggestions, not that modeling is approved or ready.
Only aggregate column metadata is inspected; inputs are not modified.
"""

__all__ = ["advise_models"]

_GOALS = ("Explore", "Classification", "Numeric prediction", "Forecasting", "Clustering")


def _count(value, default=None):
    # Booleans are not meaningful counts, despite being Python integers.
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default


def _kind(column):
    return str(column.get("kind", "")).strip().lower()


def _identifier(column):
    return _kind(column).startswith("identifier")


def _numeric(column):
    flag = column.get("is_numeric_values")
    if isinstance(flag, bool):
        return flag
    # Backwards-compatible evidence from the existing analysis.py profile.
    return _kind(column) == "numeric" and isinstance(column.get("numeric"), dict)


def _classes(column):
    return _count(column.get("category_count"), _count(column.get("distinct")))


def _present(column, rows):
    return max(0, rows - _count(column.get("missing"), 0))


def _useful(column, rows):
    return (not _identifier(column) and _kind(column) != "all blank"
            and _present(column, rows) > 0
            and _count(column.get("distinct")) != 1)


def _high_cardinality(column, rows):
    distinct = _classes(column)
    present = _present(column, rows)
    # A review flag, not proof that a column is an identifier or unusable label.
    return distinct is not None and distinct > 20 and distinct >= present / 2


def _projects(columns, rows):
    projects = []
    numeric = [c for c in columns if _useful(c, rows) and _numeric(c)]
    labels = [c for c in columns if _useful(c, rows) and not _numeric(c)
              and _kind(c).startswith("categorical")
              and (_classes(c) or 0) >= 2 and not _high_cardinality(c, rows)]
    dates = [c for c in columns if c.get("date_like") is True and not _identifier(c)]
    if columns:
        projects.append("Explore missingness, duplicates and column distributions before defining a modeling objective.")
    if labels:
        projects.append("Classification could be explored if a categorical column is deliberately chosen as a meaningful outcome and usable predictors exist.")
    if numeric:
        projects.append("Numeric prediction could be explored if a verified numerical measurement is deliberately selected as the target and predictors are available before that outcome.")
    if numeric and dates:
        projects.append("Forecasting could be explored after confirming a real timestamp, numerical outcome, observation unit, ordering and sampling frequency.")
    if numeric:
        projects.append("Clustering could explore similarity in meaningful numerical features, after scaling and a domain review of what similarity should mean.")
    return projects


def advise_models(profile, goal="Explore", target=None):
    """Return aggregate-only, conditional suggestions for an optional user goal.

    ``target`` is an exact column name, not a column index. Explore and Clustering
    never require a target. Missing optional enrichment metadata is tolerated.
    Numerical evidence does not establish the business meaning of a column.
    """
    profile = profile if isinstance(profile, dict) else {}
    columns = profile.get("columns", [])
    columns = [c for c in columns if isinstance(c, dict)] if isinstance(columns, list) else []
    rows = _count(profile.get("rows"), 0)
    result = dict(
        task=goal if isinstance(goal, str) else "Unknown goal",
        status="needs_review", candidates=[], rationale=[], evaluation=[],
        cautions=[
            "Advisory only: local rules suggest candidates; no model is trained, ranked or selected. Choice depends on the target, objective, sample size and validation design.",
            f"The profile contains {rows} rows before exclusions. Sample adequacy depends on usable independent observations, feature complexity, effect size and validation needs; no fixed row-count cutoff guarantees adequacy.",
            "With small samples, complex models and evaluation metrics can be unstable; exclusions, correlated rows or rare classes may make the proposed validation infeasible. Collect more relevant observations or simplify the objective when needed.",
            "Data alone cannot prove fairness or establish an automatic bias/unfairness verdict. Review protected groups, outcomes, use context and appropriate authorization before any fairness assessment or consequential deployment.",
            "Review privacy, consent, representativeness, missingness and duplicate/entity structure. Aggregate metadata cannot establish causality or suitability for deployment.",
        ],
        possible_projects=[],
    )
    if not isinstance(goal, str) or goal not in _GOALS:
        result["status"] = "unsupported_goal"
        result["rationale"].append("Unsupported goal. Choose exactly: " + ", ".join(_GOALS) + ".")
        return result
    if goal == "Explore":
        result["status"] = "exploration"
        result["rationale"].append("Uploading alone supports a descriptive report; a goal and target are optional. No user intent is inferred from column properties.")
        result["possible_projects"] = _projects(columns, rows)
        if target is not None:
            result["rationale"].append("The optional target is ignored in Explore mode; select a prediction goal to assess an outcome.")
        result["evaluation"].append("Review descriptive patterns and data quality with domain context; model evaluation requires a deliberately defined objective.")
        return result
    if rows < 2:
        result["status"] = "insufficient_data"
        result["rationale"].append("Fewer than two rows cannot support an independent training/validation comparison. Collect usable observations; two rows would still not establish adequacy.")
        return result
    if goal == "Clustering":
        return _clustering(result, columns, rows, target)
    if target is None or target == "":
        result["status"] = "needs_target"
        result["rationale"].append(f"{goal} requires a deliberately selected target column; the descriptive report remains available without one.")
        return result
    matches = [c for c in columns if c.get("name") == target] if isinstance(target, str) else []
    if len(matches) != 1:
        result["status"] = "invalid_target"
        result["rationale"].append("The target must match exactly one existing column name. Select a valid target; no replacement was guessed.")
        return result
    outcome = matches[0]
    if _identifier(outcome):
        result["status"] = "invalid_target"
        result["rationale"].append("The selected target is marked identifier / ambiguous. Do not predict a guessed ID; verify its meaning and correct the profile before choosing it as an outcome.")
        return result
    present = _present(outcome, rows)
    distinct = _count(outcome.get("distinct"))
    if present == 0 or _kind(outcome) == "all blank" or _classes(outcome) == 0:
        result["status"] = "invalid_target"
        result["rationale"].append("The target has no usable nonblank outcomes; a supervised objective cannot be evaluated.")
        return result
    if distinct == 1 or _classes(outcome) == 1:
        result["status"] = "invalid_target"
        result["rationale"].append("The target is constant / single-class. There is no outcome variation to learn or evaluate; review the objective or collect varied outcomes.")
        return result
    result["cautions"].append("Never encode or include the target itself among the input features. Exclude post-outcome fields and proxies that leak the answer; fit imputation, encoding, scaling and feature selection on training folds only.")
    if _count(outcome.get("missing"), 0):
        result["cautions"].append(f"Only {present} rows have nonblank target values. Do not silently impute supervised labels; investigate missing outcomes and report exclusions.")
    if present < 2:
        result["status"] = "insufficient_data"
        result["rationale"].append("Fewer than two nonblank outcomes remain; independent training/validation comparison is not possible.")
        return result
    if outcome.get("date_like") is True:
        result["status"] = "needs_review"
        result["rationale"].append("Date-like target requires semantic review: confirm meaningful class labels or a numerical measurement rather than automatically treating timestamps as outcomes. Resolve or correct the date-like interpretation before considering candidate models. For forecasting, use the date as the time axis and select a separate verified numeric target.")
        return result
    if (goal == "Classification"
            and str(outcome.get("semantic_kind", "")).strip().lower() == "mixed numeric/text"):
        result["status"] = "needs_review"
        result["rationale"].append("The target is confirmed mixed numeric/text. Resolve inconsistent tokens and confirm meaningful, consistently encoded class labels before classification; do not treat accidental numeric/text mixtures as valid classes. No candidate models are suggested until this semantic review is complete.")
        return result
    if goal in ("Numeric prediction", "Forecasting") and not _numeric(outcome):
        result["status"] = "needs_review"
        result["rationale"].append("Goal/type mismatch: a numeric outcome has not been confirmed. Verify all nonblank target values are finite numeric measurements, with meaningful units, not text labels, dates or guessed IDs.")
        return result
    if goal == "Forecasting":
        return _forecasting(result, columns, outcome)
    features = [c for c in columns if c is not outcome and _useful(c, rows)]
    if not features:
        result["status"] = "needs_review"
        result["rationale"].append("No nonconstant, non-ID predictor is apparent beyond the target. Review usable pre-outcome features before considering supervised candidates.")
        return result
    result["cautions"].append("Exclude meaningless IDs and review high-cardinality fields; encode genuine categories rather than assigning arbitrary numeric order. Group related entities and duplicates within a split; use chronological splits where the intended use predicts the future.")
    if goal == "Classification":
        return _classification(result, outcome, rows)
    result["status"] = "advisory"
    result["candidates"] = [
        "Dummy regressor (training-set mean baseline)",
        "Linear regression and Ridge regression (scaled numeric features; encoded categories)",
        "Decision tree regressor (nonlinear baseline with limited complexity)",
        "Random forest regressor (bagged nonlinear trees)",
        "Extra Trees regressor (randomized tree ensemble)",
        "Histogram gradient boosting regressor (nonlinear boosting candidate; dense encoded data may use more memory)",
    ]
    result["rationale"].append("The profile provides numeric-value evidence for the selected non-ID target. Verify it represents a measurement, not numeric class codes; define units, error costs and prediction timing before modeling.")
    result["evaluation"] = [
        "Compare MAE and RMSE in target units against the dummy baseline and a linear baseline on held-out observations; select metrics according to the real costs of errors.",
        "Use cross-validation or a held-out test set matched to deployment (grouped or chronological when appropriate). Report fold variability, residual patterns and uncertainty; do not tune on the final test set.",
    ]
    result["cautions"].append("Impute predictors and encode categories within training folds. Scale numerical predictors for regularized linear methods; random forests do not generally require scaling. More complexity is justified only by relevant validation evidence.")
    return result


def _classification(result, outcome, rows):
    classes = _classes(outcome)
    result["status"] = "advisory"
    if classes is None:
        result["status"] = "needs_review"
        result["rationale"].append("The number of target classes is unknown. Confirm discrete labels and at least two classes before considering classifiers.")
        return result
    if _high_cardinality(outcome, rows):
        result["status"] = "needs_review"
        result["rationale"].append("Potential goal/type mismatch: the target has many distinct values relative to observations. Confirm genuine discrete class labels rather than continuous measurements, free text or unrecognized IDs; no classifier is suggested until reviewed.")
        return result
    if _numeric(outcome) and classes > 2:
        result["status"] = "needs_review"
        result["rationale"].append("Potential goal/type mismatch: this numeric target must be confirmed as discrete class codes rather than a continuous measurement before classification.")
    elif not _numeric(outcome) and not _kind(outcome).startswith("categorical"):
        result["status"] = "needs_review"
        result["rationale"].append("Confirm that mixed/text target values are consistent discrete labels, not free text or inconsistent encodings.")
    else:
        result["rationale"].append("The chosen target appears to contain discrete labels. Confirm class meanings and the positive/event label where relevant; numeric labels are not automatically continuous outcomes.")
    result["candidates"] = [
        "Dummy classifier (majority-class baseline)",
        "Logistic regression (with category encoding and numerical scaling)",
        "Decision tree classifier (nonlinear rule-based model)",
        "Random forest classifier (balanced tree ensemble)",
        "Extra Trees classifier (randomized tree ensemble)",
        "Linear support vector machine (scaled sparse feature representation)",
        "K-nearest neighbors (scaled features; sensitive to feature geometry)",
        "Histogram gradient boosting classifier (nonlinear boosting candidate; dense encoded data may use more memory)",
    ]
    result["evaluation"] = [
        "Compare macro-F1, balanced accuracy, per-class precision/recall and a confusion matrix with the dummy baseline; raw accuracy alone can hide class imbalance.",
        "Use stratified validation only when minority samples permit each split to contain the required classes; otherwise collect data or revise the evaluation design. Respect entity groups/time ordering and report variability, not a single guaranteed score.",
    ]
    minority = _count(outcome.get("minority_count"))
    if minority is not None:
        result["cautions"].append(f"The smallest class has {minority} observed sample(s). Minority sample counts constrain held-out splits, fold count and reliable per-class estimates; class weights do not create new evidence.")
        if minority < 2:
            result["status"] = "needs_review"
            result["cautions"].append("A class with fewer than two examples cannot appear in both training and a held-out set. The candidates are conditional only; standard stratified evaluation is not currently supportable.")
    else:
        result["cautions"].append("Minority class count is unknown. Inspect the complete class distribution before selecting stratified folds; a top-category summary alone may omit rare classes.")
    result["cautions"].append("Inspect class imbalance and minority representation. If justified, compare class weighting or resampling only inside training folds; assess decision thresholds/calibration separately where probability decisions matter.")
    return result


def _forecasting(result, columns, outcome):
    dates = [c for c in columns if c is not outcome and c.get("date_like") is True
             and not _identifier(c)]
    result["status"] = "needs_review"
    if not dates:
        result["rationale"].append("Forecasting requires a detected date column separate from the numeric target. None is available in the supplied metadata; time_charts alone do not prove usable timestamps.")
        return result
    result["rationale"].append("A numeric target and date-like column are present. Date-like detection is a hint, not proof: confirm parsing, true chronology, timestamp meaning, observation unit, gaps, duplicate times and any multiple series before proceeding.")
    result["candidates"] = [
        "Naive last-observation baseline",
        "Moving-average baseline over recent observations",
        "Simple exponential-smoothing baseline",
        "Seasonal naive baseline (only after confirming the seasonal period)",
        "Linear regression and Ridge regression with lag and calendar features",
        "Random forest and Extra Trees with lag and calendar features",
    ]
    result["evaluation"] = [
        "Use chronological holdouts or rolling-origin backtesting with a declared forecast horizon; never random train/test splitting for forecasting.",
        "Compare MAE/RMSE against the naive baseline at the same horizon. Construct lags and preprocessing using past training data only; report errors by horizon and uncertainty when supportable.",
    ]
    result["cautions"].append("No frequency, seasonality or sufficient history is inferred from column names, date-like flags or time charts. Verify observations per period, gaps and enough independent history before seasonal/complex candidates; do not invent future exogenous features.")
    return result


def _clustering(result, columns, rows, target):
    if target is not None:
        result["rationale"].append("Clustering is unsupervised: the optional target is ignored, not encoded or used as a label. Choose input features deliberately; a label-like column must not be included by accident.")
    numeric = [c for c in columns if _useful(c, rows) and _numeric(c)
               and c.get("date_like") is not True]
    if not numeric:
        result["status"] = "needs_review"
        result["rationale"].append("No meaningful nonconstant numerical input is apparent after excluding IDs and date-like fields. Do not run distance-based clustering on arbitrary text/category codes.")
        return result
    result["status"] = "advisory"
    result["candidates"] = [
        "K-means (conditional on scaled meaningful numerical features and an appropriate distance notion)",
        "Agglomerative clustering (conditional on distance/linkage choices and sample size)",
        "DBSCAN (optional if density structure, scale and parameter sensitivity are defensible)",
    ]
    result["rationale"].append("Numerical features may support a similarity exploration, not a prediction outcome or proof that real groups exist. Select features and distance measures using domain meaning.")
    result["evaluation"] = [
        "Inspect silhouette only when a valid partition has at least two clusters and fewer clusters than samples; it is not supervised accuracy or proof of meaningful groups.",
        "Assess stability across seeds, resamples, scaling and parameters, then review clusters and noise with domain experts. A high internal score alone does not establish business usefulness.",
    ]
    result["cautions"].append("Scale meaningful numerical features; review skew, outliers and missing-data handling. Exclude meaningless IDs, high-cardinality codes and raw timestamps; arbitrary category encoding distorts distances. Do not interpret clusters as protected groups or fairness verdicts.")
    return result
