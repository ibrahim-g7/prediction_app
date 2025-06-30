# src/predict.py

import sys
import os
import joblib
import pandas as pd

def resource_path(filename_in_model_dir):
    """ Get absolute path to resource, works for dev and for PyInstaller. """
    try:
        base_bundle_dir = sys._MEIPASS
        model_folder_path = os.path.join(base_bundle_dir, "model")
    except AttributeError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_folder_path = os.path.join(script_dir, os.pardir, "model")
    return os.path.join(model_folder_path, filename_in_model_dir)


def get_prediction(input_df, raw_output=False):
    """
    Takes a DataFrame with numerical features and returns a prediction
    using the new stacked model.
    """
    try:
        # 1. Load the scaler
        scaler = joblib.load(resource_path("scaler.joblib"))
        num_cols_to_scale = joblib.load(resource_path("scaled_features_list.joblib"))

        # The input DataFrame is already numerical, so we just need to scale it.
        processed_df = input_df.copy()

        # 2. Apply scaling
        if num_cols_to_scale:
            # Ensure all columns expected by the scaler are present
            for col in num_cols_to_scale:
                if col not in processed_df.columns:
                    processed_df[col] = 0 # Safety net
            
            data_to_scale = processed_df[num_cols_to_scale]
            scaled_data = scaler.transform(data_to_scale)
            processed_df[num_cols_to_scale] = scaled_data

        # 3. Load the base models
        models = {}
        # IMPORTANT: Update this list if your base models have changed
        for model_name in ["random_forest", "xgboost", "linear_reg"]:
            models[model_name] = joblib.load(resource_path(f"{model_name}.joblib"))
        
        # 4. Load the meta model
        meta_model = joblib.load(resource_path("meta_model.joblib"))

        # 5. Make predictions with each base model
        base_predictions = {}
        for model_name, model in models.items():
            # Since all data is numerical, we can directly predict
            pred = model.predict(processed_df[model.feature_names_in_])[0]
            base_predictions[model_name] = pred

        # 6. Create meta features and make the final prediction
        meta_features_df = pd.DataFrame([base_predictions])
        final_prediction = meta_model.predict(meta_features_df[meta_model.feature_names_])[0]
        
        if raw_output:
            return final_prediction
        
        return f"{final_prediction:,.2f} AED"

    except FileNotFoundError as e:
        error_msg = f"Error: A required model file was not found. Please check your 'model' directory. Missing file: {e.filename}"
        print(error_msg)
        return error_msg
    except Exception as e:
        import traceback
        error_msg = f"An unexpected error occurred in prediction: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

