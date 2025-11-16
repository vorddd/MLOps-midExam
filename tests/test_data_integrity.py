from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_model_artifact_is_loadable():
    """Loading the trained pipeline should not raise errors."""
    artifact_path = PROJECT_ROOT / "deployment" / "best_model_pipeline.joblib"
    assert artifact_path.exists(), "Model artifact missing for deployment"

    model = joblib.load(artifact_path)
    assert hasattr(model, "predict"), "Loaded object is not a scikit-learn estimator"
