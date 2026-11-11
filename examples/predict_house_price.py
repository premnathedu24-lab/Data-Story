"""Load a trusted exported Data Story regression pipeline and predict a sample house price.

Usage: python examples/predict_house_price.py path/to/data_story_model.pkl
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to a trusted exported regression pipeline (.pkl)")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}")

    # Only load pickle artifacts from a source you trust.
    with args.model.open("rb") as f:
        model = pickle.load(f)

    new_house = pd.DataFrame([{
        "longitude": -118.48,
        "latitude": 34.24,
        "housing_median_age": 29,
        "total_rooms": 2110.5,
        "total_bedrooms": 431,
        "population": 1161,
        "households": 406,
        "median_income": 3.5475,
        "ocean_proximity": "<1H OCEAN",
    }])
    prediction = model.predict(new_house)
    print("Model loaded:", type(model))
    print("Predicted median house value:", f"${prediction[0]:,.2f}")


if __name__ == "__main__":
    main()
