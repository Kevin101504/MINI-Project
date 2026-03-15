import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(actual, predicted):

    plt.figure(figsize=(10,5))

    plt.plot(actual.values, label="Actual Demand")
    plt.plot(predicted, label="Predicted Demand")

    plt.title("Demand Forecast")
    plt.xlabel("Time")
    plt.ylabel("Demand")

    plt.legend()

    plt.show()


def plot_feature_importance(model, feature_names):

    importance = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    df = df.sort_values("importance", ascending=False)

    plt.figure(figsize=(10,5))

    plt.bar(df["feature"], df["importance"])

    plt.xticks(rotation=45)

    plt.title("Feature Importance")

    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.tight_layout()

    plt.show()