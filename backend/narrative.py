"""Prose built only from numbers and titles the investigation already holds.

Format every figure with the same rounding the client displays, and only
mention a comparison the metrics support.
"""

from __future__ import annotations

from .bench import Scored

METRIC_LABEL = {
    "average_precision": "average precision",
    "f1": "F1",
    "recall": "recall",
    "precision": "precision",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced accuracy",
    "roc_auc": "ROC AUC",
}

OPERATING = {"f1", "recall", "precision", "accuracy", "balanced_accuracy"}


def fmt(value: float) -> str:
    return f"{value:.2f}"


def rank_key(scored: Scored, metric: str, *, require_feasible: bool) -> tuple:
    feasible = 1 if scored.feasible or not require_feasible else 0
    return (feasible, scored.metrics.get(metric, float("-inf")))


def reading_for(scored: Scored, baseline: Scored | None, metric: str) -> dict:
    label = METRIC_LABEL.get(metric, metric)
    value = scored.metrics[metric]
    headline = "No baseline to compare."
    body: list[str] = []
    if baseline is None or baseline.path_id == scored.path_id:
        headline = "Reference path"
        body.append(f"{label.capitalize()} is {fmt(value)}.")
    else:
        delta = value - baseline.metrics[metric]
        if delta >= 0.02:
            headline = "Worked"
            body.append(
                f"{label.capitalize()} is {fmt(value)}, up {fmt(delta)} from the reference path ({fmt(baseline.metrics[metric])})."
            )
        elif delta <= -0.02:
            headline = "Did not work"
            body.append(
                f"{label.capitalize()} is {fmt(value)}, down {fmt(abs(delta))} from the reference path ({fmt(baseline.metrics[metric])})."
            )
        else:
            headline = "Little difference"
            body.append(
                f"{label.capitalize()} is {fmt(value)}, within 0.02 of the reference path ({fmt(baseline.metrics[metric])})."
            )
        body.append(_tradeoff(scored, baseline))
    if metric in scored.ci:
        lo, hi = scored.ci[metric]
        body.append(f"The 95% bootstrap band on {label} is {fmt(lo)}–{fmt(hi)}.")
    return {"headline": headline, "body": " ".join(body)}


def _tradeoff(scored: Scored, baseline: Scored) -> str:
    recall_delta = scored.metrics["recall"] - baseline.metrics["recall"]
    precision_delta = scored.metrics["precision"] - baseline.metrics["precision"]
    accuracy_delta = scored.metrics["accuracy"] - baseline.metrics["accuracy"]
    bits = []
    if recall_delta >= 0.05:
        bits.append(f"Recall rose from {fmt(baseline.metrics['recall'])} to {fmt(scored.metrics['recall'])}.")
    elif recall_delta <= -0.05:
        bits.append(f"Recall fell from {fmt(baseline.metrics['recall'])} to {fmt(scored.metrics['recall'])}.")
    if precision_delta <= -0.05:
        bits.append(f"Precision fell from {fmt(baseline.metrics['precision'])} to {fmt(scored.metrics['precision'])}.")
    elif precision_delta >= 0.05:
        bits.append(f"Precision rose from {fmt(baseline.metrics['precision'])} to {fmt(scored.metrics['precision'])}.")
    if accuracy_delta <= -0.03:
        bits.append(
            f"Accuracy moved the other way, from {fmt(baseline.metrics['accuracy'])} to {fmt(scored.metrics['accuracy'])}, "
            "because more ordinary rows are flagged."
        )
    if not bits:
        return "Precision, recall, and accuracy stayed close to the reference path."
    return " ".join(bits)


def conclude(
    titles: dict[str, str],
    scored: list[Scored],
    metric: str,
    *,
    precision_floor: float | None,
    reference_id: str | None,
) -> dict:
    if not scored:
        return {
            "leader_id": None,
            "leader": None,
            "paragraphs": ["No path ran, so there is nothing to rank."],
            "question": "Remove a constraint, or point the search at papers that name a method this bench can run.",
            "metric": metric,
            "ranking": [],
        }

    require = precision_floor is not None
    ordered = sorted(scored, key=lambda item: rank_key(item, metric, require_feasible=require), reverse=True)
    leader = ordered[0]
    leader_title = titles.get(leader.path_id, leader.path_id)
    label = METRIC_LABEL.get(metric, metric)
    paragraphs: list[str] = []

    if require and not leader.feasible:
        paragraphs.append(
            f"No path held precision ≥ {precision_floor:.2f} on the validation slice. "
            f"The closest is {leader_title}, and I am not calling it a success."
        )
    else:
        ci = leader.ci.get(metric)
        band = f" The test-set band on {label} is {fmt(ci[0])}–{fmt(ci[1])}." if ci else ""
        paragraphs.append(f"{leader_title} leads on {label} at {fmt(leader.metrics[metric])}.{band}")

    if precision_floor is not None and leader.feasible and leader.metrics["precision"] + 1e-9 < precision_floor:
        paragraphs.append(
            f"The cutoff for {leader_title} held the precision floor on the validation slice, "
            f"but test precision is {fmt(leader.metrics['precision'])}, under {precision_floor:.2f}. "
            "I did not move the cutoff to repair the test number."
        )

    if reference_id and reference_id != leader.path_id:
        reference = next((item for item in scored if item.path_id == reference_id), None)
        if reference is not None:
            paragraphs.append(
                f"Against the reference path, {label} moved from {fmt(reference.metrics[metric])} to {fmt(leader.metrics[metric])}."
            )

    if len(ordered) > 1:
        second = ordered[1]
        second_title = titles.get(second.path_id, second.path_id)
        lead_ci = leader.ci.get("average_precision")
        second_ci = second.ci.get("average_precision")
        if lead_ci and second_ci and metric == "average_precision":
            if lead_ci[0] > second_ci[1]:
                paragraphs.append(
                    f"The average-precision bands of {leader_title} and {second_title} do not overlap, "
                    "so this ordering is not an artifact of one draw of the test rows. "
                    "The bands do not refit the models, so training noise is still unmeasured."
                )
            else:
                paragraphs.append(
                    f"The average-precision bands of {leader_title} ({fmt(lead_ci[0])}–{fmt(lead_ci[1])}) and "
                    f"{second_title} ({fmt(second_ci[0])}–{fmt(second_ci[1])}) overlap. "
                    "I would not treat this lead as settled."
                )

    # Surface a disagreement the human might care about.
    by_accuracy = max(scored, key=lambda item: item.metrics["accuracy"])
    by_ap = max(scored, key=lambda item: item.metrics["average_precision"])
    question = "If a constraint in the margin is wrong, say so and I will re-rank from the scores I already have."
    if by_accuracy.path_id != by_ap.path_id:
        question = (
            f"Accuracy picks {titles.get(by_accuracy.path_id, by_accuracy.path_id)} "
            f"({fmt(by_accuracy.metrics['accuracy'])}). "
            f"Average precision picks {titles.get(by_ap.path_id, by_ap.path_id)} "
            f"({fmt(by_ap.metrics['average_precision'])}). "
            "Which mistake is more expensive: missing a rare event, or raising a false alarm?"
        )
    elif precision_floor is None:
        question = (
            "If false alarms are the expensive mistake, give me a precision floor "
            "(for example 0.40) and I will retune every cutoff on the validation slice."
        )

    losers = [item for item in ordered if item.path_id != leader.path_id]
    failed = []
    if reference_id:
        reference = next((item for item in scored if item.path_id == reference_id), None)
        if reference:
            for item in losers:
                if item.metrics[metric] <= reference.metrics[metric] - 0.02:
                    failed.append(titles.get(item.path_id, item.path_id))
    if failed:
        paragraphs.append("Did not beat the reference on " + label + ": " + "; ".join(failed) + ".")

    return {
        "leader_id": leader.path_id,
        "leader": leader_title,
        "metric": metric,
        "metric_label": label,
        "paragraphs": paragraphs,
        "question": question,
        "ranking": [item.path_id for item in ordered],
        "precision_floor": precision_floor,
    }


def metric_reason(counts: dict[str, int], positive_rate: float) -> tuple[str, str]:
    """Choose a ranking metric from words that actually appear in the abstracts."""
    ap = counts.get("average_precision", 0)
    f1 = counts.get("f1", 0)
    accuracy = counts.get("accuracy", 0)
    if positive_rate < 0.2 and (ap or f1) and ap + f1 >= accuracy:
        metric = "average_precision" if ap >= f1 else "f1"
        reason = (
            f"The rare class is {positive_rate:.0%} of the bench. "
            f"In the abstracts I kept, average precision or precision-recall is named {ap} times, "
            f"F1 or F-measure {f1} times, and accuracy {accuracy} times. "
            f"I am ranking by {METRIC_LABEL[metric]}."
        )
        return metric, reason
    if accuracy > ap + f1:
        return "accuracy", (
            f"Accuracy is the metric these abstracts name most often ({accuracy} mentions). "
            "On a rare-event bench that choice flatters a classifier that ignores the rare class. "
            "Say so in the margin if you want average precision instead."
        )
    metric = "average_precision" if ap >= f1 else "f1"
    if ap == f1 == 0:
        metric = "average_precision"
        return metric, (
            "The abstracts do not settle the metric. The rare class is "
            f"{positive_rate:.0%} of the bench, so I am ranking by average precision rather than accuracy. "
            "Override that in the margin if accuracy is really the decision."
        )
    return metric, (
        f"I counted metric names in the kept abstracts: average precision or precision-recall {ap}, "
        f"F1 {f1}, accuracy {accuracy}. I am ranking by {METRIC_LABEL[metric]}."
    )


def count_metric_mentions(texts: list[str]) -> dict[str, int]:
    counts = {"average_precision": 0, "f1": 0, "accuracy": 0}
    for text in texts:
        hay = text.lower()
        counts["average_precision"] += hay.count("average precision") + hay.count("precision-recall") + hay.count("pr auc") + hay.count("auprc")
        counts["f1"] += hay.count("f1") + hay.count("f-measure")
        counts["accuracy"] += hay.count("accuracy")
    return counts
