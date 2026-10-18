"""The desk may only show words and methods that came from somewhere real."""

from backend.bench import make_dataset, run_path, score_run
from backend.margin import interpret
from backend.methods import BY_ID, Path, plan_paths
from backend.narrative import conclude, fmt
from backend.sources import Paper, abstract_quotes, parse_arxiv_feed, plan_query


FEED = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>https://arxiv.org/abs/2401.01234v2</id>
    <published>2024-02-02T00:00:00Z</published>
    <title>  SMOTE for imbalanced classification  </title>
    <summary>  This paper studies SMOTE for imbalanced classification. Random oversampling is the simpler baseline.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Grace Hopper</name></author>
    <link href="https://arxiv.org/pdf/2401.01234v2" type="application/pdf"/>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>https://arxiv.org/api/errors</id>
    <title>Error</title>
    <summary>Not a paper.</summary>
  </entry>
</feed>
"""


def _paper(pid, title, abstract):
    return Paper(
        paper_id=pid,
        title=title,
        authors=["A Author"],
        published="2024-01-01",
        abstract=abstract,
        link=f"https://arxiv.org/abs/{pid}",
        pdf_url=None,
        categories=["cs.LG"],
        venue=None,
        origin="arxiv",
    )


def test_arxiv_parser_keeps_real_fields_and_strips_version():
    papers = parse_arxiv_feed(FEED)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.paper_id == "2401.01234"
    assert paper.title == "SMOTE for imbalanced classification"
    assert paper.authors == ["Ada Lovelace", "Grace Hopper"]
    assert paper.link == "https://arxiv.org/abs/2401.01234"
    assert "SMOTE" in paper.abstract
    assert paper.categories[0] == "cs.LG"


def test_quotes_are_substrings_and_invented_sentences_are_dropped():
    paper = _paper(
        "1",
        "Title",
        "We propose SMOTE for the minority class. The weather in Lisbon was unrelated and is not the method.",
    )
    quotes = abstract_quotes(paper, ["smote", "minority"])
    assert quotes
    abstract = " ".join(paper.abstract.lower().split())
    for quote in quotes:
        assert " ".join(quote.text.lower().split()) in abstract
    assert all("Lisbon" not in quote.text for quote in quotes)


def test_smote_proper_name_does_not_count_as_random_oversampling():
    text = "We use the synthetic minority oversampling technique (SMOTE)."
    assert BY_ID["smote"].matches(text)
    assert not BY_ID["random_oversample"].matches(text)
    both = "We compare random oversampling and SMOTE on the same split."
    assert BY_ID["smote"].matches(both)
    assert BY_ID["random_oversample"].matches(both)


def test_paths_combine_only_methods_the_papers_name():
    papers = [
        _paper("a", "About SMOTE", "We apply SMOTE before a classifier."),
        _paper("b", "Forests", "A random forest is our classifier."),
    ]
    paths = plan_paths(papers)
    ids = {path.path_id for path in paths}
    assert "random_forest" in ids
    assert "smote__random_forest" in ids
    assert "logreg" not in ids
    combined = next(path for path in paths if path.path_id == "smote__random_forest")
    cited = {citation.method_id for citation in combined.citations}
    assert cited == {"smote", "random_forest"}
    for citation in combined.citations:
        assert citation.quote in ("We apply SMOTE before a classifier.", "A random forest is our classifier.")

    unrelated = [_paper("c", "Transformers", "We study attention in language models and report perplexity.")]
    assert plan_paths(unrelated) == []


def test_next_releases_review_and_a_struck_paper_rebuilds_the_list():
    import asyncio

    from backend.session import Investigation

    waiting = Investigation("imbalanced classification")
    waiting.phase = "review"
    asyncio.run(waiting.handle(action={"type": "proceed"}))
    assert waiting.approved is True

    papers = Investigation("imbalanced classification")
    papers.phase = "papers"
    papers.papers = [_paper("a", "About SMOTE", "We apply SMOTE before a classifier.")]
    asyncio.run(papers.handle(action={"type": "approve_papers"}))
    assert papers.papers_ready is True
    early = Investigation("imbalanced classification")
    early.phase = "reading"
    asyncio.run(early.handle(action={"type": "approve_papers"}))
    assert early.papers_ready is False

    idle = Investigation("imbalanced classification")
    idle.phase = "done"
    asyncio.run(idle.handle(action={"type": "proceed"}))
    assert idle.approved is False

    review = Investigation("imbalanced classification")
    review.phase = "review"
    review.papers = [
        _paper("a", "About SMOTE", "We apply SMOTE before a classifier."),
        _paper("b", "Forests", "A random forest is our classifier."),
    ]
    review.paths = plan_paths(review.papers)
    assert any(path.path_id == "smote__random_forest" for path in review.paths)
    asyncio.run(review.handle(action={"type": "drop_paper", "paper_id": "a"}))
    assert "a" in review.dropped
    assert all("smote" not in path.path_id for path in review.paths)
    assert any(path.path_id == "random_forest" for path in review.paths)


def test_pivot_keeps_the_standing_direction_and_requires_the_focus():
    from backend.sources import compose_direction, select_relevant

    standing = "Which training methods help a classifier catch rare events in tabular data without drowning in false alarms?"
    combined = compose_direction(standing, "focus more on threshold selection")
    assert standing in combined
    assert "threshold selection" in combined
    again = compose_direction(combined, "focus more on threshold selection")
    assert again.lower().count("threshold selection") == 1

    plan = plan_query(combined)
    assert "all:imbalanced" in plan.query
    assert "all:threshold" in plan.query
    assert "threshold" in plan.required
    assert any("earlier direction" in note for note in plan.notes)

    kept = select_relevant(
        [
            _paper("1", "Imbalanced classification", "We study imbalanced classification and report accuracy."),
            _paper("2", "Thresholds", "We study threshold selection for imbalanced classification."),
        ],
        plan,
    )
    assert [paper.paper_id for paper in kept] == ["2"]


def test_query_expansion_is_visible():
    plan = plan_query("Which methods catch rare events without false alarms?")
    assert "all:imbalanced" in plan.query
    assert "cat:cs.LG" in plan.query
    assert any("imbalanced classification" in note for note in plan.notes)


def test_pivot_searches_keywords_and_honors_negation():
    from backend.sources import select_relevant

    direction = (
        "pivot to regular llm jailbreaing, not multimodal, and not agents, "
        "and i want to find a gap that exists and explore it"
    )
    plan = plan_query(direction)
    assert "jailbreak" in plan.query
    assert "all:llm" in plan.query
    assert " OR " in plan.query
    assert "pivot" not in plan.query
    assert "jailbreaing" not in plan.query
    assert "multimodal" in plan.excluded
    assert "agents" in plan.excluded
    assert any("jailbreaing" in note for note in plan.notes)

    kept = select_relevant(
        [
            _paper(
                "1",
                "Jailbreaking Large Language Models",
                "We study jailbreak attacks on an LLM and report the success rate of the prompts.",
            ),
            _paper(
                "2",
                "Jailbreaking Multimodal Models",
                "We jailbreak a multimodal LLM with image prompts.",
            ),
            _paper(
                "3",
                "Scaling Laws for Language Models",
                "We train a large language model and measure perplexity.",
            ),
        ],
        plan,
    )
    assert [paper.paper_id for paper in kept] == ["1"]


def test_margin_presets_map_to_one_action():
    assert interpret("No synthetic rows. Do not invent minority samples.") == [{"type": "rule", "rule": "no_synthetic"}]
    assert interpret("Rank by accuracy.") == [{"type": "metric", "metric": "accuracy"}]
    floor = interpret("I need recall, but keep precision at or above 0.40.")
    assert floor == [{"type": "precision_floor", "value": 0.4}]
    assert interpret("I wonder if the moon matters.") == []
    assert interpret("search threshold selection for imbalanced data")[0]["type"] == "redirect"


def test_conclusion_uses_only_computed_numbers():
    def scored(pid, ap, acc):
        from backend.bench import Scored

        return Scored(
            path_id=pid,
            threshold=0.5,
            threshold_rule="Cutoff 0.5.",
            feasible=True,
            metrics={
                "average_precision": ap,
                "f1": 0.2,
                "recall": 0.3,
                "precision": 0.4,
                "accuracy": acc,
                "balanced_accuracy": 0.5,
                "roc_auc": 0.6,
            },
            ci={"average_precision": [ap - 0.05, ap - 0.01], "recall": [0.2, 0.4]},
            fit_note="",
            n_pos_before=10,
            n_pos_after=10,
            n_fit_rows=20,
        )

    summary = conclude(
        {"plain": "Logistic regression", "weighted": "Class weighting, then logistic regression"},
        [scored("plain", 0.20, 0.93), scored("weighted", 0.41, 0.80)],
        "average_precision",
        precision_floor=None,
        reference_id="plain",
    )
    text = " ".join(summary["paragraphs"])
    assert summary["leader"] == "Class weighting, then logistic regression"
    assert fmt(0.41) in text
    assert "0.99" not in text
    assert "Accuracy picks" in summary["question"]


def test_data_grounding_uses_a_named_dataset_or_a_stated_synthesis():
    from backend.data_ground import ground_papers

    named = ground_papers([
        _paper(
            "1",
            "A survey",
            "We compare resampling methods on the Wisconsin breast cancer data and report F1.",
        )
    ])
    assert named is not None and named.loader == "breast_cancer"
    assert "breast cancer" in named.evidence.lower()
    diagnosis = ground_papers([
        _paper(
            "mri",
            "Improving Breast Cancer Grade Prediction with Multiparametric MRI",
            "Breast cancer was diagnosed for over 7.8 million women between 2015 to 2020. Grading plays a vital role.",
        )
    ])
    assert diagnosis is None

    from backend.data_ground import dataset_name_plan
    homeless = dataset_name_plan(
        "Train a synthetic model that predicts chronic homelessness from shelter records."
    )
    assert "all:homelessness" in homeless.query
    assert "all:chronic" in homeless.query
    assert "all:train" not in homeless.query
    assert "homelessness" in homeless.required
    assert "train" not in homeless.required

    from backend.data_ground import dataset_plan
    plan = dataset_plan("Which training methods help a classifier catch rare events?")
    assert "all:dataset" in plan.query
    assert "all:imbalanced" in plan.query

    stated = ground_papers([
        _paper(
            "2",
            "A generator",
            "We generate a synthetic set of 5000 samples and 12 features from a Gaussian distribution with an imbalance ratio of 1:9.",
        )
    ])
    assert stated is not None and stated.kind == "synthesis"
    assert stated.n_rows == 5000 and stated.n_features == 12
    assert abs(stated.positive_rate - 0.1) < 1e-9

    vague = ground_papers([
        _paper("3", "No data", "We study imbalanced classification and show that SMOTE helps logistic regression.")
    ])
    assert vague is None
    sonar_image = ground_papers([
        _paper(
            "sas",
            "Additional Representations for Improving Synthetic Aperture Sonar Classification Using Convolutional Neural Networks",
            "Object classification in synthetic aperture sonar (SAS) imagery is usually a data starved and class imbalanced problem.",
        )
    ])
    assert sonar_image is None
    sonar_table = ground_papers([
        _paper(
            "uci",
            "Mines and rocks",
            "We compare classifiers on the UCI sonar dataset of mines versus rocks.",
        )
    ])
    assert sonar_table is not None and sonar_table.loader == "sonar"
    named_yeast = ground_papers([
        _paper("4", "Yeast", "We evaluate resampling on the yeast dataset from the KEEL repository and report F1.")
    ])
    assert named_yeast is not None and named_yeast.loader == "yeast"
    from backend.data_ground import dataset_name_plan
    named_plan = dataset_name_plan("Which training methods help a classifier catch rare events?")
    assert "all:yeast" in named_plan.query
    assert "all:pima" in named_plan.query


def test_grounded_script_loads_the_named_dataset(tmp_path):
    from backend.bench import run_experiment_code, score_run
    from backend.data_ground import ground_papers

    paper = _paper(
        "bc",
        "Breast cancer baselines",
        "The experiments use the Wisconsin breast cancer table and logistic regression.",
    )
    grounding = ground_papers([paper])
    branch = Path(
        path_id="logreg",
        title="Logistic regression",
        component_ids=["logreg"],
        citations=[],
        assumptions=[],
    )
    table = tmp_path / "table.csv"
    stored, source, _script = run_experiment_code(branch, grounding=grounding, table_path=str(table))
    assert "load_breast_cancer" in source
    assert "csv.writer" in source
    assert table.is_file()
    header = table.read_text().splitlines()[0]
    assert header.endswith("label,split")
    scored = score_run(stored, components=["logreg"], precision_floor=None)
    assert 0 <= scored.metrics["average_precision"] <= 1


def test_papiers_library_hit_keeps_a_verbatim_passage():
    from backend.papiers import _paper

    paper = _paper(
        {
            "id": "doc_abc",
            "kind": "document",
            "title": "Prediction of chronic homelessness",
            "metadata": {
                "authors": ["A. Person"],
                "publishedYear": 2020,
                "identifiers": {"arxivId": "2009.09072"},
                "venue": "arXiv",
            },
            "evidence": [
                {
                    "kind": "highlight",
                    "id": "hl_1",
                    "snippet": "a dataset for 6521 individuals was generated",
                },
                {"kind": "passage", "id": "pas_2", "snippet": "this sentence was not in the paper"},
            ],
        },
        {
            "id": "doc_abc",
            "ok": True,
            "content": "Using a 30-day time step, a dataset for 6521 individuals was generated. The model forecasts chronic homelessness.",
            "source": {"family": "pdf", "url": "https://arxiv.org/pdf/2009.09072"},
        },
    )
    assert paper is not None
    assert paper.origin == "papiers"
    assert paper.paper_id == "2009.09072"
    assert [quote.text for quote in paper.quotes] == ["a dataset for 6521 individuals was generated"]
    assert "this sentence was not in the paper" not in paper.abstract


def test_github_urls_are_taken_from_the_paper_text():
    from backend.sources import github_urls

    text = "Code at https://github.com/example/method.git, and again https://github.com/example/method."
    assert github_urls(text) == ["https://github.com/example/method"]


def test_gemini_script_parser_rejects_attacks_and_keeps_code():
    from backend.gemini_agent import extract_script

    script = extract_script("```python\nprint('{\"summary\": \"ok\"}')\n```")
    assert "print" in script
    from backend.gemini_agent import _without_comments

    quiet = _without_comments(
        'x = "keep # this"\n# drop me\n"""a note"""\nprint(x)\n'
    )
    assert quiet.splitlines() == ['x = "keep # this"', "print(x)"]
    try:
        extract_script("```python\nimport subprocess\nprint('no')\n```")
    except RuntimeError as exc:
        assert "not run" in str(exc)
    else:
        raise AssertionError("attack-shaped script was accepted")


def test_question_script_counts_only_papers_that_use_every_word():
    from backend.question_experiment import run_question_script

    papers = [
        _paper("1", "Agentic MLLMs", "We build an agentic mllm that calls tools from an image."),
        _paper("2", "Language models", "We train a language model and measure perplexity."),
    ]
    payload, source = run_question_script(["agentic", "mllm"], papers)
    assert "mentions" in source
    assert payload["matched"] == 1
    assert payload["total"] == 2
    assert "1 of 2" in payload["reading"]
    assert payload["rows"][0]["all"] is True
    assert payload["rows"][1]["all"] is False


def test_branch_script_runs_and_returns_probabilities():
    from backend.bench import make_dataset, run_experiment_code, score_run

    data = make_dataset(3)
    path = Path(
        path_id="logreg",
        title="Logistic regression",
        component_ids=["logreg"],
        citations=[],
        assumptions=[],
    )
    stored, source, script = run_experiment_code(path, data)
    assert "LogisticRegression" in source
    assert "rng.normal" in source
    assert "backend.bench" not in source
    assert script.endswith("experiments/logreg.py")
    assert len(stored.p_test) == len(data.y_test)
    scored = score_run(stored, components=["logreg"], precision_floor=None)
    assert 0 <= scored.metrics["average_precision"] <= 1


def test_bench_fits_on_train_only_and_returns_scores():
    data = make_dataset(3)
    assert data.y_test.sum() > 0
    path = Path(
        path_id="logreg",
        title="Logistic regression",
        component_ids=["logreg"],
        citations=[],
        assumptions=[],
    )
    stored = run_path(path, data)
    assert len(stored.p_test) == len(data.y_test)
    assert stored.p_test.min() >= 0 and stored.p_test.max() <= 1
    scored = score_run(stored, components=["logreg"], precision_floor=None)
    assert 0 <= scored.metrics["average_precision"] <= 1
    # A precision floor must be chosen from validation scores, so changing
    # the test labels cannot be what picked the cutoff. We just check it runs.
    floored = score_run(stored, components=["logreg"], precision_floor=0.4)
    assert floored.threshold is not None


def test_branch_combines_only_cited_approaches_and_names_a_gap():
    from backend.branch import judge_joined, judge_split, plan_branches, still_grounded

    defense = _paper(
        "d",
        "A Practical Defense",
        "This paper proposes a defense against jailbreak attacks on an LLM.",
    )
    persona = _paper(
        "p",
        "Persona Prompts",
        "We study persona prompts as a jailbreak method against an LLM.",
    )
    both = _paper(
        "b",
        "Persona Defenses",
        "We test persona prompts against a defense for jailbreak attacks on an LLM.",
    )
    plan = plan_query("llm jailbreaking, not multimodal")
    branches = plan_branches([defense, persona], plan)
    split = next(branch for branch in branches if branch.kind == "split")
    assert set(split.approach_ids) == {"defense", "persona"}
    for citation in split.citations:
        source = defense.abstract if citation.paper_id == "d" else persona.abstract
        assert citation.quote in source or citation.quote in (defense.title, persona.title)

    unrelated = _paper("u", "Jailbreak Survey", "We survey jailbreak attacks on an LLM and report a taxonomy.")
    missed = judge_split(split, [unrelated], query=split.query, origin="arxiv")
    assert missed.status == "gap"
    assert missed.found == []

    fresh = plan_branches([defense, persona], plan)
    split = next(branch for branch in fresh if branch.kind == "split")
    hit = judge_split(split, [both], query=split.query, origin="arxiv")
    assert hit.status == "published"
    assert [paper.paper_id for paper in hit.found] == ["b"]
    assert hit.found[0].quotes
    assert hit.found[0].quotes[0].text in both.abstract

    joined = plan_branches([both], plan)
    assert joined and all(branch.kind == "joined" for branch in joined)
    marked = judge_joined(joined[0])
    assert marked.status == "joined"
    assert still_grounded(split, {"d"}) is False
