import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")
def predict_custom(input_dict, model_path, scaler_path, le_path, y_le_path, num_cols_path):
    """
    Loads the trained model and preprocessing objects to predict the target value
    for a single custom input dictionary.
    
    Args:
        input_dict (dict): A dictionary containing the input features.
        model_path (str): The file path to the saved PyTorch model state.
        scaler_path (str): The file path to the saved StandardScaler.
        le_path (str): The file path to the saved feature LabelEncoders.
        y_le_path (str): The file path to the saved target LabelEncoder.
        num_cols_path (str): The file path to the saved numerical columns list.
        
    Returns:
        list: top 3 predicted crop types.
    """
    # Define model parameters
    num_features = 15
    num_classes = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load preprocessing objects
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(le_path)
    y_le = joblib.load(y_le_path)
    num_cols = joblib.load(num_cols_path)
    class FTTransformer(nn.Module):
        def __init__(self, num_features, num_classes, d_model=64, num_heads=4, num_layers=3, dropout=0.1):
            super().__init__()
            
            self.input_layer = nn.Linear(num_features, d_model)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dropout=dropout,
                activation='gelu'
            )
            
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Final linear layer for classification (outputs a logit for each class)
            self.fc_out = nn.Linear(d_model, num_classes)

        def forward(self, x):
            x = self.input_layer(x)
            x = x.unsqueeze(0)
            x = self.transformer(x)
            x = x.squeeze(0)
            return self.fc_out(x)
    # Re-initialize the model with the correct architecture and load the saved state
    model = FTTransformer(num_features, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create a DataFrame from the input dictionary
    df_new = pd.DataFrame([input_dict])

    # Identify categorical and numerical columns from the loaded data
    cat_cols_loaded = list(label_encoders.keys())
    all_cols = num_cols + cat_cols_loaded
    missing_cols = set(all_cols) - set(df_new.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input_dict: {missing_cols}")

    df_new = df_new[all_cols]

    # Encode categorical features
    for col in cat_cols_loaded:
        if col in df_new.columns:
            le = label_encoders[col]
            try:
                df_new[col] = le.transform(df_new[col].astype(str))
            except ValueError:
                print(f"Warning: Category '{df_new[col].iloc[0]}' for column '{col}' not seen during training. Setting to -1.")
                df_new[col] = -1

    # Scale numerical features
    df_new[num_cols] = scaler.transform(df_new[num_cols])

    # Convert the preprocessed DataFrame to a PyTorch tensor
    X_new = torch.tensor(df_new.values, dtype=torch.float32).to(device)

    # Perform prediction
    with torch.no_grad():
        preds_logits = model(X_new)
        # Get the top 3 predicted class indices
        top3_preds = torch.topk(preds_logits, 3, dim=1).indices.cpu().numpy()[0]
    
    # Decode the top 3 predicted class indices back to the original string labels
    predicted_crops = y_le.inverse_transform(top3_preds)
    return predicted_crops.tolist()

# Example of a custom input dictionary.
# Note that "Crop Type" is not included as it is the target variable to be predicted.
custom_input = {
    "Region": "South",
    "Soil Type": "Sandy",
    "Rainfall_mm": 1250,
    "Temperature(c)": 30,
    "Fertilizer_Used": True,
    "Irrigation_Used": True,
    "Weather_Condition": "Rainy",
    "Days_to_Harvest": 185,
    "Humidity": 88,
    "Moisture": 75,
    "Nitrogen": 35,
    "Potassium": 5,
    "Phosphorous": 22,
    "Fertilizer Name": "DAP",
    "Yield_tons_per_hectare" : 7.0
}
model_path = 'ft_transformer_model.pth'
scaler_path = 'scaler.joblib'
le_path = 'label_encoders.joblib'
y_le_path = 'y_le.joblib'
num_cols_path = 'num_cols.joblib'
# The predict_custom function will now load the necessary files to make the prediction
try:
    prediction = predict_custom(custom_input, model_path, scaler_path, le_path, y_le_path, num_cols_path)
    print(f"\nPredicted Crop Type: {prediction}")
except ValueError as e:
    print(f"Error: {e}")