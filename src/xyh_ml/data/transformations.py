import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xyh_ml.config.data import DataConfig

# -----------------------------------------------------------------------------
# Z score scaling (calculation and application)
# -----------------------------------------------------------------------------


def get_z_score_variables(
    data_config: DataConfig,
):
    return [
        v.name
        for v in data_config.input_variables + data_config.input_parameters
        if v.transformation == "z_score"
    ]


def setup_z_score_scaler(
    df: pd.DataFrame,
    data_config: DataConfig,
):
    # Get the list of input variables to be transformed with Z score
    input_z_score = get_z_score_variables(data_config)
    standard_scaler = StandardScaler()
    standard_scaler.fit(df[input_z_score])

    return standard_scaler


def apply_z_score_scaler(
    df: pd.DataFrame,
    standard_scaler: StandardScaler,
    data_config: DataConfig,
) -> pd.DataFrame:
    # Get the list of input variables to be transformed with Z score
    input_z_score = get_z_score_variables(data_config)

    # Apply Z score transformation to the input variables that require it
    df[input_z_score] = standard_scaler.transform(df[input_z_score])

    return df


# -----------------------------------------------------------------------------
# One-hot encoding
# -----------------------------------------------------------------------------


def get_classes(
    data_config: DataConfig,
):
    return [c.name for c in data_config.output_classes]


def setup_one_hot_encoder(
    df: pd.DataFrame,
    data_config: DataConfig,
):
    # Fit the one-hot encoder for target classes
    one_hot_columns = ["class"]
    target_classes = get_classes(data_config)
    classes_encoder = OneHotEncoder(categories=[target_classes])
    classes_encoder.fit(df[one_hot_columns])

    return classes_encoder


def apply_one_hot_encoder(
    df: pd.DataFrame,
    classes_encoder: OneHotEncoder,
) -> pd.DataFrame:
    # Apply the one-hot encoder to the target classes
    one_hot_columns = ["class"]
    df.loc[:, classes_encoder.get_feature_names_out(one_hot_columns)] = (
        classes_encoder.transform(df[one_hot_columns]).toarray()
    )

    return df


# -----------------------------------------------------------------------------
# Set up and apply all transformations
# -----------------------------------------------------------------------------


def transform(
    dfs: dict[tuple[str, str], pd.DataFrame],
    data_config: DataConfig,
):
    # Get splits from the input dictionary
    splits = set(k[0] for k in dfs.keys())

    # Container for scaled and encoded datasets
    dfs_trans = {}

    # Store the fitted scalers and encoders for each split
    transforms = {}

    # Set up the transformations (fit the scalers and encoders) for each split
    # by fitting them on the training data
    for split in splits:
        # Fit the Z score scaler and one-hot encoder for the training data
        df = dfs[(split, "train")]
        standard_scaler = setup_z_score_scaler(df, data_config)
        classes_encoder = setup_one_hot_encoder(df, data_config)
        transforms.setdefault(split, []).extend(
            [standard_scaler, classes_encoder]
        )

        # Apply the transformations to all datasets
        for key in (k for k in dfs.keys() if k[0] == split):
            df = dfs[key].copy()
            df = apply_z_score_scaler(df, standard_scaler, data_config)
            df = apply_one_hot_encoder(df, classes_encoder)
            dfs_trans[key] = df

    return dfs_trans, transforms
