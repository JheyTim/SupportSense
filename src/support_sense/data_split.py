import pandas as pd
from sklearn.model_selection import train_test_split

# Use one stable seed for reproducible project experiments.
RANDOM_STATE = 42


def split_dataset(
    dataframe: pd.DataFrame, target_column: str = "category"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split labeled data into stratified train, validation, and test sets."""

    # First reserve 30% of the complete dataset.
    train_df, temporary_df = train_test_split(
        dataframe,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=dataframe[target_column],
    )

    # The temporary set represents 30% of the original data.
    # Dividing it in half produces 15% validation and 15% test sets.
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary_df[target_column],
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, validation_df, test_df
