"""Train and persist the winning P06 model artifact for API inference."""

from __future__ import annotations

from ml_app.api.model import DEFAULT_ARTIFACT_PATH, train_and_save_winning_model


def main() -> None:
    artifact_path = train_and_save_winning_model()
    print(f"Winning model artifact generated at: {artifact_path}")
    if artifact_path != DEFAULT_ARTIFACT_PATH:
        print(f"Default artifact path is: {DEFAULT_ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
