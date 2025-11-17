from pathlib import Path

import joblib
import pandas as pd

from deployment import prediction

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DummyModel:
    def predict(self, features):
        return [1] * len(features)

def test_load_model_exists():
    assert hasattr(prediction, "load_model")
    
def test_shipping_dataset_has_expected_columns():
    """Ensure the raw dataset is present and includes the expected signals."""
    dataset_path = PROJECT_ROOT / "shipping.csv"
    assert dataset_path.exists(), "shipping.csv is missing from the repository"

    data = pd.read_csv(dataset_path)
    expected_columns = {
        "ID",
        "Warehouse_block",
        "Mode_of_Shipment",
        "Customer_care_calls",
        "Customer_rating",
        "Cost_of_the_Product",
        "Prior_purchases",
        "Product_importance",
        "Gender",
        "Discount_offered",
        "Weight_in_gms",
        "Reached.on.Time_Y.N",
    }

    assert expected_columns.issubset(
        set(data.columns)
    ), "Dataset schema changed unexpectedly"
    assert len(data) > 0, "Dataset is empty"


def test_load_model_returns_predictable_estimator(monkeypatch, tmp_path):
    """load_model should load either the local artifact or download fallback."""

    dummy_path = tmp_path / "dummy_model.joblib"
    joblib.dump(_DummyModel(), dummy_path)

    if not prediction.LOCAL_MODEL_PATH.exists():
        monkeypatch.setattr(prediction, "LOCAL_MODEL_PATH", dummy_path)

    # Ensure the cache does not return a stale object between tests.
    if hasattr(prediction.load_model, "clear"):
        prediction.load_model.clear()

    model = prediction.load_model()
    assert hasattr(model, "predict"), "Model missing predict method"
