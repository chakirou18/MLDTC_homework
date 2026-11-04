import pickle 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score,mean_squared_error

#parameters
output_file  = 'model_rfr'

#Data preparation 
df = pd.read_csv("./energydata_complete.csv")
pd.set_option('display.max_columns', None)
df.isnull().sum()
duplicates = df[df.duplicated(keep=False)]

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract hour and minutes
hour = df['date'].dt.hour + df['date'].dt.minute / 60

# Create cyclical hour features
df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

#Split the data:
# Split the data
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42
)

df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42
)

# Define features and target
features = [
    'lights', 'T1', 'T2', 'T3',
    'T4', 'T5', 'T6', 'T7',
    'T8', 'T9', 'T_out', 'RH_1',
    'RH_2', 'RH_3', 'RH_4', 'RH_5',
    'RH_6', 'RH_7', 'RH_8', 'RH_9',
    'RH_out', 'Press_mm_hg', 'Windspeed',
    'Visibility', 'Tdewpoint',
    'hour_sin', 'hour_cos'
]

target = 'Appliances'

# Define target values
y_train = df_train[target].values
y_test = df_test[target].values
y_true = df_val[target].values


# training 
def Rtrain(df_train, y_train,n=300, r=42):
    X_train = df_train[features]
    rf = RandomForestRegressor(n_estimators=n, max_depth=None, random_state=r)
    rf.fit(X_train, y_train)
    return rf

# Make a prediction
def Rpredict(df, rf):
    X = df[features]
    y_pred = rf.predict(X)
    return y_pred

# validation
print(f'doing validation with r')

rf = Rtrain(df_train, y_train)
y_val_pred = Rpredict(df_val, rf)
mae = mean_absolute_error(y_true, y_val_pred)
rmse = root_mean_squared_error(y_true, y_val_pred)
r2 = r2_score(y_true, y_val_pred)
print("Random forest")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# Training the final model
print('Training the final model')

y_full_train = df_full_train[target].values

rf = Rtrain(df_full_train, y_full_train)

# Make prediction on test data
y_test_pred = Rpredict(df_test, rf)

# Evaluate
mae = mean_absolute_error(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Final Random Forest model")
print("MAE(Wh):", mae)
print("RMSE(Wh):", rmse)
print("R²:", r2)

# Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump(rf, f_out)

print(f'the model is saved to {output_file}')