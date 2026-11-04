# EnergyPredict ml — Energy Consumption Prediction Using Machine Learning

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

### 6.2 Python Environment and Dependencies

The project was tested on Linux (GitHub Codespaces) with **Python 3.11.17**. It uses a dedicated virtual environment (`.venv`) so that its dependencies are separate from other projects.

From the `Energy-prediction` directory, create and activate the environment:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

Install the dependencies listed in `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

The following core versions were verified during development:

| Dependency | Tested version |
| --- | --- |
| NumPy | 2.3.3 |
| Pandas | 3.1.0 |
| Scikit-learn | 1.7.2 |
| Flask | 3.1.3 |

The requirements file also includes Matplotlib, Seaborn, XGBoost, Requests and ipykernel for notebook analysis and testing. Their exact versions are not yet pinned in the current requirements file.

Verify that the packages are available:

```bash
python -c "import numpy, pandas, sklearn, flask, matplotlib, seaborn, xgboost, requests, ipykernel; print('All libraries installed successfully')"
python -c "import sklearn; print(sklearn.__version__)"
```

The commands above were executed successfully in the project's `.venv` environment, and the Scikit-learn version was confirmed as **1.7.2**. Make sure `.venv/` is included in `.gitignore`; the virtual environment itself should not be committed to Git.

**Important:** The saved Pickle model should be loaded with the same Scikit-learn version used during training. Only load Pickle files from trusted sources.

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

With the virtual environment activated and `energydata_complete.csv` in the project directory, run:

```bash
python train.py
```

The script creates `hour_sin` and `hour_cos` from the `date` column, and trains the selected **RandomForestRegressor** (`n_estimators=300`, `max_depth=None`, `random_state=42`). It uses the following random data split: 60% training, 20% validation and 20% test. After validation, it retrains the final model on the combined 80% training and validation data, evaluates it on the 20% test set and saves the model as **`model_rfr`**.

**Results observed on 4 November 2026** (using this dataset and configuration):

| Metric | Validation model | Final model on test set |
| --- | ---: | ---: |
| MAE (Wh) | 33.0187 | 29.5608 |
| RMSE (Wh) | 68.3423 | 62.9250 |
| R² | 0.5346 | 0.6043 |

These are results from a random split of time-ordered measurements; they do not establish performance on future periods or on other buildings. The validation and test metrics also correspond to models trained on different amounts of data.

### 6.5 Run and Test the Local Flask Prediction API

Start the API in a terminal, from the project directory with `.venv` activated:

```bash
python predict.py
```

The Flask development server listens on port **9696**. In a second terminal, use the example JSON input file `house.json` (it must be present in the project directory) to send a request:

```bash
curl -i -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  -d @house.json
```

**Verified local API test (4 November 2026):**

```text
HTTP/1.1 200 OK
Content-Type: application/json

{
  "Energy": 152.13333333333333
}
```

The API accepts environmental measurements and a date/time in JSON, creates the cyclical hour features, loads the previously trained `model_rfr` and returns the estimated appliance energy consumption in Wh. The example prediction above was produced by the tested model; values may change after retraining.

This verifies a **local Flask API**, not a public cloud deployment. The Flask debug server is for development only and should be replaced with a production WSGI server for deployment.

### 6.6 Current Status and Next Steps

- Completed: EDA, model comparison and hyperparameter experiments in the notebook.
- Completed: standalone `train.py`, saved model `model_rfr`, and successful local `/predict` HTTP request.
- Completed: project `.venv`, `requirements.txt` and tested dependency installation.
- Planned: production WSGI setup, user-facing HTML/CSS form, Docker containerization and cloud deployment.

To reproduce the API test from a fresh clone, include `house.json` as a non-sensitive example input (or provide an equivalent sample JSON), and ensure the `model_rfr` file is present or generated by `python train.py`.

### 6.7 Dataset Attribution

Candanedo, L. (2017). *Appliances Energy Prediction*. UCI Machine Learning Repository.
## 7. Docker Containerization

EnergyPredict AI is containerized using Docker. The Docker image includes Python 3.11, the required dependencies, the trained Random Forest model, Flask, and Gunicorn.

### 7.1 Build the Docker Image

From the `Energy-prediction` project directory, run:

```bash
docker build -t energypredict-ml:latest .
```

### 7.2 Run the Docker Container

```bash
docker run --rm --name energypredict-ml -p 9696:9696 energypredict-ml:latest
```

The application runs inside the container using Gunicorn and listens on port `9696`.

### 7.3 Test the Prediction API

Open another terminal in the project directory and run:

```bash
curl -i -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  -d @house.json
```

Example response:

```json
{
  "Energy": 152.13333333333333
}
```

The HTTP response status should be `200 OK`.

The example predicts household appliance energy consumption in Wh for a 10-minute measurement interval.

### 7.4 Docker Files

- `Dockerfile`: defines the Docker image and starts the Flask application using Gunicorn.
- `.dockerignore`: excludes unnecessary files from the build context.
- `requirements.txt`: contains the Python dependencies.
- `predict.py`: provides the prediction API.
- `model_rfr`: contains the trained Random Forest model.
- `house.json`: provides example input data for testing.

### 7.5 Validation

The Docker image was built and tested successfully on November 4, 2026.

The `/predict` endpoint returned HTTP `200 OK` and a prediction of approximately `152.13 Wh` using the example input data.

DOI: https://doi.org/10.24432/C5VC8G

