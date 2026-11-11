import unittest
import pandas as pd
import numpy as np

from ml_engine import (
    train_model,
    train_forecast,
    train_clustering,
    load_frame,
    _date_ratio
)


class MLEngineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 120
        rng = np.random.default_rng(7)

        cls.df = pd.DataFrame({
            'date': pd.date_range(
                '2025-01-01', periods=n
            ).astype(str),

            'orders': np.arange(n) % 17 + 5,
            'ad_spend': np.arange(n) * 1.7 + 20,

            'channel': np.where(
                np.arange(n) % 3 == 0,
                'Web',
                np.where(
                    np.arange(n) % 3 == 1,
                    'Email',
                    'Referral'
                )
            ),

            'segment': np.where(
                np.arange(n) % 2 == 0,
                'New',
                'Returning'
            ),

            'revenue': (
                np.arange(n) * 13.5
                + rng.normal(0, 12, n)
            ),
        })

        cls.data = cls.df.to_csv(index=False).encode()

    # --------------------------------------------------
    # REGRESSION TEST
    # --------------------------------------------------

    def test_regression_trains_compares_and_predicts(self):
        r = train_model(
            self.data,
            'revenue',
            'regression',
            ['orders', 'ad_spend', 'channel', 'segment']
        )

        self.assertIn(r['best_model'], r['results'])
        self.assertIn('RMSE', r['best_metrics'])
        self.assertEqual(
            len(r['predictions']),
            r['test_rows']
        )

    # --------------------------------------------------
    # CLASSIFICATION TEST
    # --------------------------------------------------

    def test_classification_trains_and_confusion_matrix(self):
        r = train_model(
            self.data,
            'segment',
            'classification',
            ['orders', 'ad_spend', 'channel']
        )

        self.assertIn('F1 (weighted)', r['best_metrics'])
        self.assertIsNotNone(r['confusion_matrix'])

    # --------------------------------------------------
    # FORECASTING TEST - HORIZON 1
    # --------------------------------------------------

    def test_forecasting_uses_chronological_split_and_returns_future(self):
        r = train_forecast(
            self.data,
            'revenue',
            'date',
            lags=5,
            horizon=1
        )

        self.assertEqual(r['task'], 'forecasting')
        self.assertEqual(len(r['future_forecast']), 1)
        self.assertEqual(r['forecast_horizon'], 1)
        self.assertIn('RMSE', r['best_metrics'])

    # --------------------------------------------------
    # FORECASTING TEST - HORIZON 5
    # --------------------------------------------------

    def test_forecasting_respects_horizon_5(self):
        r = train_forecast(
            self.data,
            'revenue',
            'date',
            lags=5,
            horizon=5
        )

        self.assertEqual(len(r['future_forecast']), 1)
        self.assertEqual(r['forecast_horizon'], 5)

        # Forecast timestamp should be five observations ahead.
        last_date = pd.to_datetime(self.df['date'].iloc[-1])
        forecast_date = pd.to_datetime(
            r['future_forecast']['Date'].iloc[0]
        )

        self.assertEqual(
            forecast_date,
            last_date + pd.Timedelta(days=5)
        )

    # --------------------------------------------------
    # FEATURE AND DATE DETECTION TEST
    # --------------------------------------------------

    def test_numeric_features_are_not_misclassified_as_dates(self):
        self.assertEqual(
            _date_ratio(self.df['orders']),
            0.0
        )

        self.assertEqual(
            _date_ratio(self.df['ad_spend']),
            0.0
        )

        self.assertGreaterEqual(
            _date_ratio(self.df['date']),
            0.8
        )

        r = train_model(
            self.data,
            'revenue',
            'regression'
        )

        self.assertIn('orders', r['features'])
        self.assertIn('ad_spend', r['features'])
        self.assertNotIn('date', r['features'])

    # --------------------------------------------------
    # NULLABLE CATEGORICAL FEATURES TEST
    # --------------------------------------------------

    def test_titanic_style_nullable_categories_train_without_pd_na_error(self):
        n = 240
        rng = np.random.default_rng(23)

        sex = np.where(
            np.arange(n) % 2 == 0,
            'female',
            'male'
        ).astype(object)

        embarked = np.where(
            np.arange(n) % 3 == 0,
            'S',
            np.where(
                np.arange(n) % 3 == 1,
                'C',
                'Q'
            )
        ).astype(object)

        age = rng.normal(30, 12, n)
        fare = rng.uniform(5, 100, n)

        survived = (
            (sex == 'female') | (fare > 60)
        ).astype(int)

        sex[::19] = None
        embarked[::13] = None
        age[::11] = np.nan

        frame = pd.DataFrame({
            'PassengerId': np.arange(1, n + 1),
            'Survived': survived,
            'Pclass': np.tile([1, 2, 3], n // 3),
            'Sex': pd.Series(sex, dtype='string'),
            'Age': age,
            'SibSp': np.arange(n) % 4,
            'Parch': np.arange(n) % 3,
            'Ticket': [
                f'TICKET-{i % 130}' for i in range(n)
            ],
            'Fare': fare,
            'Cabin': pd.Series(
                [
                    pd.NA if i % 4 else f'C{i % 30}'
                    for i in range(n)
                ],
                dtype='string'
            ),
            'Embarked': pd.Series(
                embarked,
                dtype='string'
            ),
            'Name': [
                f'Passenger {i}' for i in range(n)
            ],
        })

        raw = frame.to_csv(index=False).encode()

        r = train_model(
            raw,
            'Survived',
            'classification',
            [
                'PassengerId',
                'Pclass',
                'Sex',
                'Age',
                'SibSp',
                'Parch',
                'Ticket',
                'Fare',
                'Cabin',
                'Embarked',
                'Name'
            ]
        )

        self.assertNotIn('PassengerId', r['features'])
        self.assertNotIn('Name', r['features'])
        self.assertIn('Accuracy', r['best_metrics'])
        self.assertIsNotNone(r['confusion_matrix'])

    # --------------------------------------------------
    # FORECASTING BASELINES AND MAPE TEST
    # --------------------------------------------------

    def test_forecasting_includes_mape_and_linear_baselines(self):
        r = train_forecast(
            self.data,
            'revenue',
            'date',
            lags=5,
            horizon=1
        )

        self.assertIn(
            'Linear Regression (lags + calendar)',
            r['results']
        )

        self.assertIn(
            'MAPE (%)',
            r['best_metrics']
        )

    # --------------------------------------------------
    # MODEL LIBRARY TEST
    # --------------------------------------------------

    def test_model_library_is_expanded(self):
        clf = train_model(
            self.data,
            'segment',
            'classification',
            ['orders', 'ad_spend', 'channel']
        )

        self.assertTrue({
            'Decision Tree',
            'Extra Trees',
            'Linear SVM',
            'K-Nearest Neighbors',
            'Gradient Boosting'
        }.issubset(clf['results']))

        self.assertIn(
            'ROC-AUC',
            clf['results']['Linear SVM']
        )

        reg = train_model(
            self.data,
            'revenue',
            'regression',
            ['orders', 'ad_spend', 'channel']
        )

        self.assertTrue({
            'Linear Regression',
            'Decision Tree',
            'Extra Trees',
            'Gradient Boosting'
        }.issubset(reg['results']))

    # --------------------------------------------------
    # CSV DELIMITER TEST
    # --------------------------------------------------

    def test_ml_loader_detects_semicolon_delimiter(self):
        frame = load_frame(
            b'age;job;subscribed\n'
            b'31;admin;yes\n'
            b'42;services;no\n'
        )

        self.assertEqual(
            list(frame.columns),
            ['age', 'job', 'subscribed']
        )

        self.assertEqual(len(frame), 2)

    # --------------------------------------------------
    # TARGET CEILING WARNING TEST
    # --------------------------------------------------

    def test_target_ceiling_warning_is_generated(self):
        frame = pd.DataFrame({
            'feature': np.random.default_rng(3).normal(size=100),
            'target': np.r_[
                np.arange(90),
                np.repeat(100, 10)
            ]
        })

        r = train_model(
            frame.to_csv(index=False).encode(),
            'target',
            'regression',
            ['feature']
        )

        self.assertTrue(
            any(
                'maximum target value' in w
                for w in r['target_warnings']
            )
        )

    # --------------------------------------------------
    # CROSS-VALIDATION TEST
    # --------------------------------------------------

    def test_optional_cross_validation_is_reported(self):
        r = train_model(
            self.data,
            'segment',
            'classification',
            ['orders', 'ad_spend', 'channel'],
            run_cv=True
        )

        self.assertTrue(r['cross_validation'])
        self.assertIn(
            '3-fold',
            r['cross_validation_method']
        )

    # --------------------------------------------------
    # HOURLY FORECASTING TEST
    # --------------------------------------------------

    def test_hourly_forecasting_combines_date_and_hour(self):
        n = 240

        frame = pd.DataFrame({
            'dteday': pd.date_range(
                '2025-01-01',
                periods=n // 24
            ).repeat(24).astype(str),

            'hr': np.tile(np.arange(24), n // 24),

            'cnt': (
                100
                + np.sin(
                    np.arange(n) * 2 * np.pi / 24
                ) * 20
                + np.arange(n) * 0.05
            )
        })

        r = train_forecast(
            frame.to_csv(index=False).encode(),
            'cnt',
            'dteday',
            lags=24,
            horizon=1,
            time_column='hr',
            seasonal_period=24
        )

        self.assertEqual(
            r['duplicate_timestamps_aggregated'],
            0
        )

        self.assertEqual(
            r['rows_used'],
            n - 24
        )

        self.assertIn(
            'Seasonal Naive',
            r['results']
        )

        self.assertTrue(
            r['predictions']['Date'].str.contains(':').all()
        )

        self.assertEqual(
            len(r['future_forecast']),
            1
        )

    # --------------------------------------------------
    # CLUSTERING TEST
    # --------------------------------------------------

    def test_clustering_trains_and_returns_assignments(self):
        n = 120
        rng = np.random.default_rng(19)

        frame = pd.DataFrame({
            'x': np.r_[
                rng.normal(-2, 0.2, n // 2),
                rng.normal(2, 0.2, n // 2)
            ],

            'y': np.r_[
                rng.normal(-2, 0.2, n // 2),
                rng.normal(2, 0.2, n // 2)
            ],

            'record_id': np.arange(n)
        })

        r = train_clustering(
            frame.to_csv(index=False).encode(),
            features=['x', 'y', 'record_id'],
            n_clusters=2
        )

        self.assertIn('K-Means', r['results'])
        self.assertEqual(
            len(r['cluster_assignments']),
            n
        )

        self.assertNotIn(
            'record_id',
            r['features']
        )

    # --------------------------------------------------
    # IDENTIFIER TARGET TEST
    # --------------------------------------------------

    def test_identifier_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identifier'):
            train_model(
                self.df.assign(
                    id=np.arange(len(self.df))
                ).to_csv(index=False).encode(),
                'id',
                'regression'
            )

    # --------------------------------------------------
    # SEQUENTIAL INDEX FEATURE TEST
    # --------------------------------------------------

    def test_sequential_index_feature_is_excluded(self):
        frame = self.df.copy()
        frame['Unnamed: 0'] = np.arange(len(frame))

        r = train_model(
            frame.to_csv(index=False).encode(),
            'revenue',
            'regression',
            ['orders', 'Unnamed: 0']
        )

        self.assertNotIn(
            'Unnamed: 0',
            r['features']
        )

        self.assertTrue(
            any(
                x['feature'] == 'Unnamed: 0'
                for x in r['excluded_features']
            )
        )

    # --------------------------------------------------
    # INVALID TARGET TEST
    # --------------------------------------------------

    def test_invalid_target_rejected(self):
        with self.assertRaises(ValueError):
            train_model(
                self.data,
                'missing',
                'regression'
            )


if __name__ == '__main__':
    unittest.main()