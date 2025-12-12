# Crop Yield Prediction using FTTransformer

This project aims to predict crop yields based on various environmental and soil factors using a Feature Tokenizer Transformer (FTTransformer) model. It includes data cleaning, model training, and an inference script.

## Project Structure

- **`Data_cleaning_process.ipynb`**: Jupyter notebook containing the steps for cleaning and preprocessing the raw data (`crop_yield.csv` and `data_core.csv`). It handles merging, renaming, and type conversion to create the training dataset.
- **`final_codes.ipynb`**: Jupyter notebook for training the FTTransformer model. It includes data loading, model architecture definition, training loop, and evaluation.
- **`F_T_model.py`**: A Python script designed for inference. It loads the trained model and preprocessing artifacts (scalers, encoders) to predict crop yield/type for custom inputs.
- **`crop_yield.csv` & `data_core.csv`**: Raw datasets used for the project.
- **`ft_transformer_model.pth`**: Verified PyTorch model checkpoint.
- **`*.joblib`**: Saved preprocessing objects (scalers, label encoders) required for inference.

## Prerequisites

To run the code, you need to have Python installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- torch
- matplotlib
- joblib

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Processing
Open and run `Data_cleaning_process.ipynb` to process the raw CSV files. This will generate the clean dataset used for training.

### 2. Model Training
Open and run `final_codes.ipynb` to train the FTTransformer model. The notebook will save the trained model as `ft_transformer_model.pth` and necessary preprocessing objects as `.joblib` files.

### 3. Inference
Use the `F_T_model.py` script to make predictions on new data. The script defines a `predict_custom` function that takes a dictionary of input features and returns the predicted crop type.

Example usage within `F_T_model.py`:

```python
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
# ... calls predict_custom ...
```

Run the script:
```bash
python F_T_model.py
```

## Model Architecture
The project utilizes an **FTTransformer** (Feature Tokenizer Transformer), which is effective for tabular data. It tokenizes numerical and categorical features and processes them through Transformer encoder layers to capture complex interactions.
