"""Load a trusted exported classification pipeline and predict a sample.

Usage: python examples/predict_breast_cancer.py path/to/model.pkl
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to a trusted exported classification pipeline (.pkl)")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}")

    # Only load pickle artifacts from a source you trust.
    with args.model.open("rb") as f:
        model = pickle.load(f)

    sample = pd.DataFrame([{
        "radius_mean": 13.37, "texture_mean": 18.84, "perimeter_mean": 86.24,
        "area_mean": 551.1, "smoothness_mean": 0.0959, "compactness_mean": 0.0926,
        "concavity_mean": 0.0615, "concave points_mean": 0.0335, "symmetry_mean": 0.1792,
        "fractal_dimension_mean": 0.0615, "radius_se": 0.3242, "texture_se": 1.108,
        "perimeter_se": 2.287, "area_se": 24.53, "smoothness_se": 0.0064,
        "compactness_se": 0.0204, "concavity_se": 0.0259, "concave points_se": 0.0109,
        "symmetry_se": 0.0187, "fractal_dimension_se": 0.0032, "radius_worst": 14.97,
        "texture_worst": 25.41, "perimeter_worst": 97.66, "area_worst": 686.5,
        "smoothness_worst": 0.1313, "compactness_worst": 0.2119, "concavity_worst": 0.2267,
        "concave points_worst": 0.0999, "symmetry_worst": 0.310, "fractal_dimension_worst": 0.0761,
    }])
    prediction = model.predict(sample)
    print("Model loaded:", type(model))
    print("Predicted diagnosis:", prediction[0])
    if hasattr(model, "predict_proba"):
        print("Class probabilities:")
        for label, probability in zip(model.classes_, model.predict_proba(sample)[0]):
            print(f"{label}: {probability:.4f}")


if __name__ == "__main__":
    main()
