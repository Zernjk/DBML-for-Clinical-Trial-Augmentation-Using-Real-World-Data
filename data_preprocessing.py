import pandas as pd
import numpy as np

def preprocess_data(data, binary_indices=[], ordinal_indices=[], continuous_indices=[],
                    categorical_indices_with_levels={}):
    """
    Splits and transforms the input data into binary, ordinal, and continuous components.
    Categorical variables are one-hot encoded with (levels - 1) dimensions (to avoid redundancy).
    Even if some levels are missing in the data, the full (levels - 1) columns are preserved.

    Parameters:
    - data: pd.DataFrame or np.ndarray
    - binary_indices: list of column indices considered binary
    - ordinal_indices: list of column indices considered ordinal
    - continuous_indices: list of column indices considered continuous
    - categorical_indices_with_levels: dict {index: num_levels} — categorical to binary

    Returns:
    - dict with keys ['binary', 'ordinal', 'continuous']
    - metadata dictionary for reference, including the processed-column indices
      of genuine binary variables and each categorical dummy block
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    n = len(data)

    processed = {'binary': [], 'ordinal': [], 'cont': []}
    metadata = {
        'categorical_mappings': {},
        'column_names': list(data.columns),
        # These are indices in processed['binary'], not indices in the source data.
        'genuine_binary_indices': list(range(len(binary_indices))),
        'categorical_groups': []
    }

    for col in binary_indices:
        processed['binary'].append(data.iloc[:, col])

    for col in ordinal_indices:
        processed['ordinal'].append(data.iloc[:, col])

    for col in continuous_indices:
        processed['cont'].append(data.iloc[:, col])

    for col, levels in categorical_indices_with_levels.items():
        col_data = data.iloc[:, col]

        if levels < 2:
            raise ValueError(f"Categorical column {col} must have at least 2 levels.")

        # Use pd.Categorical to enforce category codes
        categories = list(range(levels))
        cat = pd.Categorical(col_data, categories=categories)
        one_hot = pd.get_dummies(cat, prefix=f'cat{col}', drop_first=True)  # drop first to get (levels - 1)

        # Ensure all (levels - 1) columns are present
        expected_cols = [f'cat{col}_{i}' for i in range(1, levels)]
        for expected in expected_cols:
            if expected not in one_hot.columns:
                one_hot[expected] = 0
        one_hot = one_hot[expected_cols]  # ensure column order

        group_start = len(processed['binary'])
        for subcol in one_hot.columns:
            processed['binary'].append(one_hot[subcol])
        metadata['categorical_mappings'][col] = one_hot.columns.tolist()
        metadata['categorical_groups'].append({
            'source_index': col,
            'num_levels': levels,
            'processed_indices': list(range(group_start, group_start + levels - 1)),
            'dummy_columns': one_hot.columns.tolist(),
            'reference_level': 0
        })

    for key in processed:
        if processed[key]:
            processed[key] = np.column_stack(processed[key]).reshape(n, -1)
        else:
            processed[key] = np.empty((len(data), 0))

    return processed, metadata
