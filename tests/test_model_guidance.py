"""Synthetic tests; run with python -B -m unittest test_model_guidance -v."""
import copy
import unittest

from model_guidance import advise_models


def column(name, kind="numeric", distinct=8, missing=0, **extra):
    item = dict(name=name, kind=kind, distinct=distinct, missing=missing,
                numeric={"count": 40 - missing, "minimum": 1, "maximum": 8}
                if kind == "numeric" else None)
    item.update(extra)
    return item


def sample(rows=40, columns=None, **extra):
    if columns is None:
        columns = [column("amount"), column("age", distinct=10),
                   column("label", "categorical / text", distinct=2,
                          category_count=2, minority_count=12)]
    return dict(rows=rows, columns=columns, **extra)


def text(result):
    return " ".join(result["rationale"] + result["evaluation"] + result["cautions"]).lower()


class ModelGuidanceTests(unittest.TestCase):
    def test_exact_public_shape_and_string_lists(self):
        expected = {"task", "status", "candidates", "rationale", "evaluation", "cautions", "possible_projects"}
        for goal in ("Explore", "Classification", "Numeric prediction", "Forecasting", "Clustering"):
            result = advise_models(sample(), goal, "label" if goal == "Classification" else "amount")
            self.assertEqual(set(result), expected)
            self.assertIsInstance(result["task"], str)
            self.assertIsInstance(result["status"], str)
            for field in expected - {"task", "status"}:
                self.assertIsInstance(result[field], list)
                self.assertTrue(all(isinstance(value, str) for value in result[field]))

    def test_default_upload_alone_is_exploration(self):
        result = advise_models(sample())
        self.assertEqual(result["task"], "Explore")
        self.assertEqual(result["status"], "exploration")
        self.assertEqual(result["candidates"], [])
        self.assertIn("no user intent is inferred", text(result))
        self.assertIn("optional", text(result))

    def test_explore_ignores_target_even_if_invalid(self):
        result = advise_models(sample(), target="does not exist")
        self.assertEqual(result["status"], "exploration")
        self.assertIn("ignored", text(result))

    def test_explore_projects_conditioned_on_columns(self):
        plain = advise_models(sample(columns=[column("memo", "text / mixed", distinct=40)]))
        self.assertEqual(len(plain["possible_projects"]), 1)
        rich = sample()
        rich["columns"].append(column("when", "text / mixed", distinct=40, date_like=True))
        possibilities = " ".join(advise_models(rich)["possible_projects"]).lower()
        for goal in ("classification", "numeric prediction", "forecasting", "clustering"):
            self.assertIn(goal, possibilities)

    def test_identifier_alone_does_not_prompt_numeric_project(self):
        result = advise_models(sample(columns=[column("id", "identifier / ambiguous (skipped)",
                                                        is_numeric_values=True)]))
        self.assertEqual(len(result["possible_projects"]), 1)

    def test_minimal_and_malformed_profiles_are_safe(self):
        for profile in ({}, None, {"rows": None, "columns": None},
                        {"rows": "40", "columns": [None, 4, {}]}):
            self.assertEqual(advise_models(profile)["status"], "exploration")
            advise_models(profile, "Clustering")

    def test_unsupported_goal_is_not_silently_reinterpreted(self):
        for goal in ("classification", "Predict", None, ["Explore"]):
            result = advise_models(sample(), goal)
            self.assertEqual(result["status"], "unsupported_goal")
            self.assertEqual(result["candidates"], [])

    def test_supervised_goals_require_explicit_target(self):
        for goal in ("Classification", "Numeric prediction", "Forecasting"):
            for target in (None, ""):
                result = advise_models(sample(), goal, target)
                self.assertEqual(result["status"], "needs_target")
                self.assertEqual(result["candidates"], [])

    def test_invalid_and_nonstring_targets(self):
        for target in ("absent", "Amount", 0, ["amount"]):
            result = advise_models(sample(), "Numeric prediction", target)
            self.assertEqual(result["status"], "invalid_target")
            self.assertEqual(result["candidates"], [])

    def test_ambiguous_duplicate_name_is_not_selected(self):
        result = advise_models(sample(columns=[column("x"), column("x")]), "Classification", "x")
        self.assertEqual(result["status"], "invalid_target")

    def test_identifier_target_is_blocked_even_if_numeric(self):
        p = sample(columns=[column("id", "identifier / ambiguous (skipped)",
                                   is_numeric_values=True), column("x")])
        for goal in ("Classification", "Numeric prediction", "Forecasting"):
            result = advise_models(p, goal, "id")
            self.assertEqual(result["status"], "invalid_target")
            self.assertIn("identifier", text(result))
            self.assertEqual(result["candidates"], [])

    def test_constant_and_single_class_targets(self):
        for target_column in (column("y", distinct=1),
                              column("y", "categorical / text", distinct=7, category_count=1)):
            for goal in ("Classification", "Numeric prediction", "Forecasting"):
                result = advise_models(sample(columns=[target_column, column("x")]), goal, "y")
                self.assertEqual(result["status"], "invalid_target")
                self.assertIn("single-class", text(result))

    def test_all_blank_target_is_invalid(self):
        p = sample(columns=[column("y", "all blank", distinct=0, missing=40), column("x")])
        result = advise_models(p, "Numeric prediction", "y")
        self.assertEqual(result["status"], "invalid_target")
        self.assertEqual(result["candidates"], [])

    def test_classification_candidates_and_preprocessing(self):
        result = advise_models(sample(), "Classification", "label")
        self.assertEqual(result["status"], "advisory")
        candidates = " ".join(result["candidates"]).lower()
        self.assertIn("logistic regression", candidates)
        self.assertIn("random forest classifier", candidates)
        self.assertIn("encoding", candidates)
        self.assertIn("scaling", candidates)
        for phrase in ("macro-f1", "balanced accuracy", "confusion matrix", "class imbalance"):
            self.assertIn(phrase, text(result))

    def test_rare_class_limits_are_explicit(self):
        p = sample()
        p["columns"][-1]["minority_count"] = 1
        result = advise_models(p, "Classification", "label")
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("cannot appear in both training", text(result))
        self.assertIn("conditional only", text(result))

    def test_unknown_minority_count_does_not_use_top_counts_as_complete(self):
        p = sample()
        del p["columns"][-1]["minority_count"]
        p["columns"][-1]["category_counts"] = [{"value": "major", "count": 39, "percent": 97.5}]
        result = advise_models(p, "Classification", "label")
        self.assertIn("minority class count is unknown", text(result))
        self.assertIn("may omit rare classes", text(result))

    def test_numeric_multiclass_needs_semantic_review(self):
        result = advise_models(sample(), "Classification", "amount")
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("goal/type mismatch", text(result))
        self.assertIn("discrete class codes", text(result))

    def test_numeric_binary_classification_is_possible(self):
        p = sample(columns=[column("y", distinct=2, minority_count=10), column("x")])
        result = advise_models(p, "Classification", "y")
        self.assertEqual(result["status"], "advisory")
        self.assertIn("positive/event label", text(result))

    def test_high_cardinality_target_not_assumed_to_be_classes(self):
        p = sample(columns=[column("y", distinct=38, category_count=38), column("x")])
        result = advise_models(p, "Classification", "y")
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["candidates"], [])
        self.assertIn("continuous measurements", text(result))

    def test_mixed_text_target_needs_review(self):
        p = sample(columns=[column("y", "text / mixed", distinct=5), column("x")])
        result = advise_models(p, "Classification", "y")
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("consistent discrete labels", text(result))

    def test_regression_numeric_baselines_metrics_and_leakage(self):
        result = advise_models(sample(), "Numeric prediction", "amount")
        self.assertEqual(result["status"], "advisory")
        candidates = " ".join(result["candidates"]).lower()
        for phrase in ("dummy regressor", "linear regression", "random forest regressor"):
            self.assertIn(phrase, candidates)
        for phrase in ("mae", "rmse", "never encode or include the target", "training folds only"):
            self.assertIn(phrase, text(result))

    def test_regression_text_target_is_goal_type_mismatch(self):
        result = advise_models(sample(), "Numeric prediction", "label")
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["candidates"], [])
        self.assertIn("goal/type mismatch", text(result))

    def test_explicit_numeric_false_overrides_kind(self):
        p = sample()
        p["columns"][0]["is_numeric_values"] = False
        result = advise_models(p, "Numeric prediction", "amount")
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["candidates"], [])

    def test_positive_numeric_flag_supports_old_kind_after_review(self):
        p = sample(columns=[column("y", "text / mixed", is_numeric_values=True), column("x")])
        self.assertEqual(advise_models(p, "Numeric prediction", "y")["status"], "advisory")

    def test_only_target_or_only_id_predictors_need_review(self):
        for extra in ([], [column("id", "identifier / ambiguous (skipped)")],
                      [column("constant", distinct=1)]):
            result = advise_models(sample(columns=[column("y")] + extra), "Numeric prediction", "y")
            self.assertEqual(result["status"], "needs_review")
            self.assertEqual(result["candidates"], [])

    def test_missing_outcomes_are_not_silently_imputed(self):
        p = sample()
        p["columns"][0]["missing"] = 7
        result = advise_models(p, "Numeric prediction", "amount")
        self.assertIn("33 rows", text(result))
        self.assertIn("do not silently impute", text(result))

    def test_one_usable_outcome_is_insufficient(self):
        p = sample(columns=[column("y", missing=39), column("x")])
        result = advise_models(p, "Numeric prediction", "y")
        self.assertEqual(result["status"], "insufficient_data")

    def test_date_like_classification_target_needs_review_without_candidates(self):
        for kind, numeric in (("numeric", True), ("categorical / text", False)):
            with self.subTest(kind=kind):
                p = sample(columns=[column("y", kind, distinct=2, date_like=True,
                                           is_numeric_values=numeric), column("x")])
                result = advise_models(p, "Classification", "y")
                self.assertEqual(result["status"], "needs_review")
                self.assertEqual(result["candidates"], [])
                self.assertIn("date-like target requires semantic review", text(result))

    def test_date_like_numeric_prediction_target_needs_review_without_candidates(self):
        p = sample(columns=[column("y", date_like=True, is_numeric_values=True), column("x")])
        result = advise_models(p, "Numeric prediction", "y")
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["candidates"], [])
        self.assertIn("rather than automatically treating timestamps as outcomes", text(result))

    def test_forecasting_does_not_use_date_like_numeric_target(self):
        p = sample(columns=[column("y", date_like=True, is_numeric_values=True),
                            column("when", "text / mixed", date_like=True), column("x")])
        result = advise_models(p, "Forecasting", "y")
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["candidates"], [])
        self.assertIn("select a separate verified numeric target", text(result))
        # A separate numeric measurement with a date-like time axis still works.
        valid = advise_models(p, "Forecasting", "x")
        self.assertTrue(valid["candidates"])
        self.assertIn("hint, not proof", text(valid))

    def test_confirmed_mixed_numeric_text_classification_requires_review(self):
        for kind in ("categorical / text", "text / mixed", "numeric"):
            with self.subTest(kind=kind):
                p = sample(columns=[column("y", kind, distinct=3,
                                           semantic_kind="mixed numeric/text"), column("x")])
                result = advise_models(p, "Classification", "y")
                self.assertEqual(result["status"], "needs_review")
                self.assertEqual(result["candidates"], [])
                self.assertIn("resolve inconsistent tokens", text(result))
                self.assertIn("confirm meaningful", text(result))

    def test_date_and_mixed_target_guards_do_not_block_upload_only_exploration(self):
        p = sample(columns=[column("y", date_like=True,
                                   semantic_kind="mixed numeric/text"), column("x")])
        self.assertEqual(advise_models(p)["status"], "exploration")
        self.assertEqual(advise_models(p, target="y")["status"], "exploration")

    def test_forecasting_requires_numeric_target(self):
        p = sample()
        p["columns"].append(column("date", "text / mixed", date_like=True))
        result = advise_models(p, "Forecasting", "label")
        self.assertEqual(result["candidates"], [])
        self.assertIn("goal/type mismatch", text(result))

    def test_time_charts_alone_do_not_confirm_dates(self):
        p = sample(time_charts=[{"name": "made-up-time"}])
        result = advise_models(p, "Forecasting", "amount")
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["candidates"], [])
        self.assertIn("requires a detected date column", text(result))

    def test_date_hint_yields_only_conditional_forecasting(self):
        p = sample()
        p["columns"].append(column("when", "text / mixed", date_like=True))
        result = advise_models(p, "Forecasting", "amount")
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("naive", " ".join(result["candidates"]).lower())
        for phrase in ("hint, not proof", "chronological", "never random", "no frequency, seasonality", "forecast horizon"):
            self.assertIn(phrase, text(result))

    def test_identifier_date_hint_is_not_accepted(self):
        p = sample()
        p["columns"].append(column("date_id", "identifier / ambiguous (skipped)", date_like=True))
        result = advise_models(p, "Forecasting", "amount")
        self.assertEqual(result["candidates"], [])

    def test_clustering_ignores_invalid_target(self):
        without = advise_models(sample(), "Clustering")
        with_target = advise_models(sample(), "Clustering", "not a column")
        self.assertEqual(with_target["status"], "advisory")
        self.assertEqual(with_target["candidates"], without["candidates"])
        self.assertIn("target is ignored", text(with_target))

    def test_clustering_evaluation_is_not_supervised_accuracy(self):
        result = advise_models(sample(), "Clustering")
        for phrase in ("scale meaningful numerical", "silhouette", "stability", "domain experts", "not supervised accuracy", "high-cardinality"):
            self.assertIn(phrase, text(result))

    def test_clustering_cannot_use_ids_constants_dates_or_text_alone(self):
        p = sample(columns=[column("id", "identifier / ambiguous (skipped)", is_numeric_values=True),
                            column("fixed", distinct=1), column("date", date_like=True),
                            column("words", "categorical / text", distinct=3)])
        result = advise_models(p, "Clustering")
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["candidates"], [])

    def test_fewer_than_two_rows_is_not_a_trainable_case(self):
        for rows in (0, 1):
            for goal in ("Classification", "Numeric prediction", "Forecasting", "Clustering"):
                result = advise_models(sample(rows=rows), goal, "amount")
                self.assertEqual(result["status"], "insufficient_data")
                self.assertEqual(result["candidates"], [])
        # There is deliberately no magic minimum-N gate for all useful modeling.
        self.assertEqual(advise_models(sample(rows=8), "Numeric prediction", "amount")["status"], "advisory")

    def test_safety_and_sample_adequacy_not_certifications(self):
        for goal in ("Explore", "Classification", "Numeric prediction", "Forecasting", "Clustering"):
            result = advise_models(sample(), goal, "label" if goal == "Classification" else "amount")
            for phrase in ("no model is trained", "no fixed row-count cutoff", "data alone cannot prove fairness", "protected groups", "appropriate authorization"):
                self.assertIn(phrase, text(result))
            self.assertNotIn("best model", text(result))

    def test_profile_is_not_modified_and_results_are_fresh(self):
        p = sample()
        original = copy.deepcopy(p)
        result = advise_models(p, "Classification", "label")
        result["cautions"].append("not persistent")
        self.assertEqual(p, original)
        self.assertNotIn("not persistent", advise_models(p, "Classification", "label")["cautions"])


if __name__ == "__main__":
    unittest.main()
