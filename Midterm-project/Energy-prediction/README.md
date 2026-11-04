# EnergyPredict AI — Energy Consumption Prediction Using Machine Learning

## 1. Problem Description

Energy consumption is an important challenge for households and building managers. Understanding how electrical appliances consume energy can help identify consumption patterns, improve energy monitoring, and support better energy management decisions.

However, appliance energy consumption can vary depending on several factors, including indoor temperature, humidity, outdoor weather conditions, and the time of day.

**The main problem addressed by this project is:**

*How can Machine Learning estimate the energy consumption of household appliances using environmental sensor measurements and time-related information?*

## 2. Project Objective

The objective of **EnergyPredict AI** is to develop and deploy a Machine Learning model that estimates household appliance energy consumption.

The project uses the **Appliances Energy Prediction dataset** from the UCI Machine Learning Repository, which contains 19,735 observations collected every 10 minutes.

The target variable is `Appliances`, representing energy consumption in watt-hours (Wh).

The model uses several input features, including:

- Indoor and outdoor temperatures.
- Indoor and outdoor humidity.
- Atmospheric pressure and wind speed.
- Weather-related measurements.
- Lighting energy consumption.
- Hour of the day, represented by cyclical features.

Three regression algorithms were evaluated: Ridge Regression, Random Forest Regressor, and XGBoost Regressor.

Random Forest was selected as the most promising model based on validation performance.

## 3. How the Solution Will Be Used

The final solution will be deployed as a web application connected to a Machine Learning prediction API.

Users will be able to:

1. Enter environmental sensor measurements and the corresponding time.
2. Submit the information through the web interface.
3. Receive an estimated appliance energy consumption value in Wh.
4. Use the prediction to better understand energy consumption under different conditions.

For example, a user could enter temperature, humidity, and weather measurements for a specific time. EnergyPredict AI would then estimate the appliance consumption associated with those conditions.

The initial application estimates consumption for the same measurement interval. It does not forecast the next day's consumption.

## 4. Potential Applications

EnergyPredict AI could serve as a foundation for:

- **Smart buildings:** monitoring and understanding energy consumption patterns.
- **Energy management:** supporting data-driven decisions about appliance usage.
- **IoT systems:** integrating environmental sensors with Machine Learning models.
- **Energy research:** exploring relationships between environmental conditions and electricity consumption.

The current version is a demonstration of a Machine Learning solution. Further testing with new buildings and real-world sensor data would be necessary before using it for operational energy management.

## 5. Expected Outcome

The expected result is an end-to-end Machine Learning application that includes data analysis, model training, hyperparameter tuning, model evaluation, and deployment.

The final model will be accessible through an API and a simple web interface, demonstrating how Machine Learning can be integrated into an energy monitoring application.

## 6. Reproducibility

This section explains how to obtain the dataset and reproduce the Machine Learning experiments.

### 6.1 Dataset Download

The dataset used in this project is publicly available from the UCI Machine Learning Repository.

- **Dataset name:** Appliances Energy Prediction
- **Source:** UCI Machine Learning Repository
- **Dataset page:** https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- **Direct download:** https://archive.ics.uci.edu/static/public/374/appliances%2Benergy%2Bprediction.zip
- **CSV filename:** `energydata_complete.csv`
- **Number of observations:** 19,735
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0)

**Download instructions:**

1. Open the official dataset link above.
2. Click **Download**.
3. Extract the downloaded ZIP archive.
4. Locate the file `energydata_complete.csv`.
5. Place the CSV file in the project directory, or update the dataset path in the notebook and training script accordingly.

### 6.2 Install Dependencies

Install the required Python libraries using the project's `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 6.3 Reproduce the Notebook

Open `code-source.ipynb` in Jupyter Notebook or VS Code.

Verify that the dataset path points to `energydata_complete.csv`, then execute all notebook cells in order.

The notebook covers:

- Exploratory Data Analysis (EDA).
- Data preparation and feature engineering.
- Training of Ridge, Random Forest, and XGBoost regressors.
- Hyperparameter tuning.
- Evaluation using MAE, RMSE, and R².
- Final model selection.

### 6.4 Reproduce Model Training

Once the standalone training script is available, run:

```bash
python train.py
```

The script should load the dataset, reproduce the selected preprocessing steps, train the final Random Forest model, and save the trained model for future predictions.

The dataset path, dependency versions, and training parameters must match those documented in the project to ensure reproducibility.

### 6.5 Dataset Attribution

Candanedo, L. (2017). *Appliances Energy Prediction*. UCI Machine Learning Repository.

DOI: https://doi.org/10.24432/C5VC8G

