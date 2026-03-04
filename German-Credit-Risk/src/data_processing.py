def load_data(file_path):
    """Load data from a specified file path."""
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Clean the data by handling missing values and outliers."""
    # Example cleaning steps
    data = data.dropna()  # Remove missing values
    # Additional cleaning steps can be added here
    return data

def transform_data(data):
    """Transform the data for modeling."""
    # Example transformation steps
    # This could include encoding categorical variables, scaling, etc.
    return data

def save_processed_data(data, output_path):
    """Save the processed data to the specified output path."""
    data.to_csv(output_path, index=False)