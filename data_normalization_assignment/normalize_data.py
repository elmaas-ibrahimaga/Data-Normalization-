import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Function to get the appropriate normalization method
def get_normalization_method(data_type):
    if data_type == "numerical":
        # For simplicity, return "standard_scale"
        return "standard_scale"
    return None

# Function to apply StandardScaler
def standard_scale(data):
    scaler = StandardScaler()
    return scaler.fit_transform(np.array(data).reshape(-1, 1)).flatten()

# Function to apply MinMaxScaler
def min_max_scale(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(np.array(data).reshape(-1, 1)).flatten()

# Function to apply Log Transformation
def log_transform(data):
    return np.log(np.array(data) + 1)  

# Main function to normalize data
def normalize_data(input_data):
    # Extract numerical values
    numerical_values = [feature["value"] for feature in input_data if feature["type"] == "numerical"]
    
    # Get normalization method
    method = get_normalization_method("numerical")
    
    # Apply the chosen normalization method
    if method == "standard_scale":
        normalized_values = standard_scale(numerical_values)
    elif method == "min_max_scale":
        normalized_values = min_max_scale(numerical_values)
    elif method == "log_transform":
        normalized_values = log_transform(numerical_values)
    else:
        normalized_values = numerical_values
    
    # Update input data with normalized values
    normalized_data = []
    index = 0
    for feature in input_data:
        if feature["type"] == "numerical":
            normalized_feature = feature.copy()
            normalized_feature["value"] = normalized_values[index]
            normalized_data.append(normalized_feature)
            index += 1
        else:
            normalized_data.append(feature)
    
    return normalized_data

# Sample Dataset provided
input_data = [
    {"value": 123456, "type": "numerical"},
    {"value": 500, "type": "numerical"},
    {"value": "Some Text", "type": "text"}  # Non-numerical data ignored
]

# Normalize the data
normalized_data = normalize_data(input_data)
print(normalized_data)
