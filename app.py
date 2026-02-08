import streamlit as st
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


class CorrelationSelector(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.18):
        self.threshold = threshold


    def fit(self, X, y):

        X_df = pd.DataFrame(X)

        corr = X_df.apply(lambda col: np.corrcoef(col, y)[0,1])

        self.selected_features_ = corr[
            corr.abs() >= self.threshold
        ].index.tolist()

        print("Selected Features:", len(self.selected_features_))

        return self


    def transform(self, X):

        X_df = pd.DataFrame(X)

        return X_df[self.selected_features_].values



# ------------------------------
# Load model and all input columns
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load(
        r"C:\Users\harsh\Downloads\ML_ CARS\car_price_pipeline.pkl"
    )

@st.cache_resource
def load_all_columns():
    return joblib.load(
        r"C:\Users\harsh\Downloads\ML_ CARS\input_columns.pkl"
    )

model = load_model()
all_columns = load_all_columns()


# ------------------------------
# Columns we WANT user input for
# ------------------------------
USER_INPUT_COLUMNS = [
    "Brand",
    "Body.Type",
    "Fuel.Type",
    "Gearbox.Type",
    "Drivetrain",
    "Power.hp",
    "Displacement.l",
    "Torque.lbft",
    "Cylinders",
    "MPG.City",
    "MPG.Highway",
    "Seats",
    "Doors"
]

# ------------------------------
# Categorical options
# ------------------------------
categorical_options = {
    "Brand": ['honda','bmw','lexus','hyundai','toyota','kia','nissan','audi',
              'chevrolet','ford','mercedes','porsche','infiniti','jaguar',
              'cadillac','land rover','jeep','volkswagen','maserati','subaru',
              'dodge','mazda','chrysler','aston martin','ferrari','bentley',
              'rolls royce','mclaren','lincoln','alfa romeo','volvo','mini',
              'fiat','acura','genesis','buick','gmc','lotus','suzuki','mg',
              'skoda','jac','changan'],

    "Body.Type": ['sedan','suv','hatchback','coupe','convertible',
                  'wagon','truck','van','unknown'],

    "Fuel.Type": ['petrol','diesel','hybrid','unknown'],
    "Gearbox.Type": ['automatic','manual','cvt'],
    "Drivetrain": ['front','rear','all','unknown']
}

# ------------------------------
# Numeric ranges
# ------------------------------
numeric_ranges = {
    "Power.hp": (103.0, 986.0),
    "Displacement.l": (1.2, 9.8),
    "Torque.lbft": (99.0, 1034.0),
    "Cylinders": (3.0, 12.0),
    "MPG.City": (16.0, 55.0),
    "MPG.Highway": (10.0, 58.0),
    "Seats": (2.0, 12.0),
    "Doors": (2.0, 5.0)
}

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction")
st.markdown("Provide **only important inputs**. Others are handled by the model.")

user_input = {}

for col in USER_INPUT_COLUMNS:
    if col in categorical_options:
        user_input[col] = st.selectbox(col, categorical_options[col])

    elif col in numeric_ranges:
        min_val, max_val = numeric_ranges[col]
        user_input[col] = st.slider(col, min_val, max_val, float(min_val))

# ------------------------------
# Build full input row
# ------------------------------
full_input = {col: np.nan for col in all_columns}
full_input.update(user_input)

input_df = pd.DataFrame([full_input])

# ------------------------------
# Predict
# ------------------------------
if st.button("Predict Price 💰"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Car Price: **${prediction:,.2f}**")
    except Exception as e:
        st.error("Prediction failed due to pipeline mismatch.")
        st.exception(e)
