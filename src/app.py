# src/app/app.py

import datetime
import math
import io
import json
import os 
import pandas as pd
from flask import Flask, render_template, request, send_file
from .predict import get_prediction

app = Flask(__name__)


# --- START: MODIFIED CODE TO FIND DATA FILE ---
# Get the absolute path of the directory where this script is located (i.e., .../src/)
_current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (.../)
_project_root = os.path.abspath(os.path.join(_current_dir, os.pardir))
# Build the full, robust path to the CSV file
_metro_csv_path = os.path.join(_project_root, "data", "metro_locations.csv")

# --- Load Metro Data ---
try:
    metro_df = pd.read_csv(_metro_csv_path)
    # Ensure lat/lon are numeric
    metro_df['latitude'] = pd.to_numeric(metro_df['latitude'], errors='coerce')
    metro_df['longitude'] = pd.to_numeric(metro_df['longitude'], errors='coerce')
    metro_df.dropna(subset=['latitude', 'longitude'], inplace=True)
except FileNotFoundError:
    print(f"FATAL ERROR: Metro data not found at '{_metro_csv_path}'")
    metro_df = pd.DataFrame()

# --- Helper Functions ---
def haversine(lon1, lat1, lon2, lat2):
    """Calculate the distance between two points on Earth."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Radius of Earth in kilometers
    return c * r

def find_closest_metro(user_lat, user_lon):
    """Find the closest metro station from the DataFrame."""
    if metro_df.empty:
        return None
    
    distances = metro_df.apply(
        lambda row: haversine(user_lon, user_lat, row['longitude'], row['latitude']),
        axis=1
    )
    closest_idx = distances.idxmin()
    return metro_df.loc[closest_idx]

def calculate_y_axis_range(predictions):
    """Calculates a 'nice' Y-axis range for the chart."""
    min_val, max_val = min(predictions), max(predictions)
    if min_val == max_val:
        padding = abs(min_val * 0.1) if min_val != 0 else 1
        return min_val - padding, max_val + padding
    
    order_of_magnitude = math.floor(math.log10(max_val))
    rounding_unit = 10**(order_of_magnitude - 1)
    y_min = math.floor(min_val / rounding_unit) * rounding_unit
    y_max = math.ceil(max_val / rounding_unit) * rounding_unit
    
    if y_min == y_max:
        y_min -= rounding_unit
        y_max += rounding_unit
    return y_min, y_max

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    """Handles the main page and projection generation."""
    projection_data = None
    form_data = {}
    closest_metro_name = None

    if request.method == "POST":
        try:
            form_data = request.form.to_dict()
            user_lat = float(form_data.get("latitude", 25.2048))
            user_lon = float(form_data.get("longitude", 55.2708))
            
            # Find the closest metro station
            closest_metro = find_closest_metro(user_lat, user_lon)
            if closest_metro is None:
                raise ValueError("Could not find closest metro. Check data file.")
            
            closest_metro_name = closest_metro['name']
            
            # Prepare the base DataFrame for prediction
            base_input_data = {
                "area_name_en": [closest_metro['name']],
                "rooms_en": [int(form_data.get("rooms", 1))],
                "latitude": [user_lat],
                "longitude": [user_lon],
                "latitude_metro": [closest_metro['latitude']],
                "longitude_metro": [closest_metro['longitude']],
            }
            base_df = pd.DataFrame(base_input_data)

            # Project for the next 3 years
            current_year = datetime.date.today().year
            years_to_predict = range(current_year, current_year + 4)
            predictions = []

            for year in years_to_predict:
                iter_df = base_df.copy()
                iter_df["year"] = year
                pred_value = get_prediction(iter_df, raw_output=True)
                predictions.append(pred_value)
            
            y_axis_min, y_axis_max = calculate_y_axis_range(predictions)

            projection_data = {
                "labels": [str(y) for y in years_to_predict],
                "values": predictions,
                "y_min": y_axis_min,
                "y_max": y_axis_max
            }

        except Exception as e:
            projection_data = {"error": f"Error: {str(e)}"}

    return render_template(
        "index.html",
        projection_data=projection_data,
        form_data=form_data,
        closest_metro_name=closest_metro_name
    )

@app.route("/download_excel", methods=["POST"])
def download_excel():
    """Creates and serves an Excel file from projection data."""
    try:
        data_str = request.form.get('projection_data')
        if not data_str:
            return "Error: No data received.", 400
        
        data = json.loads(data_str)
        df = pd.DataFrame({
            'Year': data['labels'],
            'Projected Price (AED)': data['values']
        })
        
        # Create an in-memory Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Projection')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='price_projection.xlsx'
        )
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)