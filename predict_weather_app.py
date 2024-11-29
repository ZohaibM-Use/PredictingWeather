from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Function to calculate percentage difference
def pct_diff(old, new):
    return (new - old) / old

# Function to compute rolling averages and percentage differences
def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather

# Function to calculate expanding mean by month and day of the year
def expand_mean(df):
    return df.expanding(1).mean()

# Load and clean the weather data
def load_weather_data():
    weather = pd.read_csv('weather.csv', index_col='DATE')
    null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]
    valid_columns = weather.columns[null_pct < .05]
    weather = weather[valid_columns].copy()
    weather.columns = weather.columns.str.lower()
    weather = weather.ffill()
    weather.index = pd.to_datetime(weather.index)
    weather['target'] = weather.shift(-1)['tmax']
    weather = weather.ffill()
    return weather

# Function to predict the weather on a given date
def predict_weather_on_date(weather, model, date_str, predictors):
    date = pd.to_datetime(date_str)

    if date.year < 1970:
        return "Error: Please enter a date after 1970."

    if date in weather.index:
        prediction = model.predict(weather.loc[[date], predictors])
        return prediction[0]

    month = date.month
    day = date.day
    year = date.year
    previous_year_data = pd.DataFrame()

    while previous_year_data.empty:
        year -= 1
        previous_year_data = weather[weather.index.year == year]
        day_data = previous_year_data[(previous_year_data.index.month == month) & 
                                       (previous_year_data.index.day == day)]
        if not day_data.empty:
            model.fit(day_data[predictors], day_data['target'])
            prediction = model.predict(day_data[predictors].values.reshape(1, -1))
            return prediction[0]
        else:
            previous_year_data = pd.DataFrame() 

# Function to predict the weather for the next seven days
def predict_next_seven_days(weather, model, date_str, predictors):
    date = pd.to_datetime(date_str)
    predictions = []
    for i in range(1, 8):
        next_date = date + pd.Timedelta(days=i)
        prediction = predict_weather_on_date(weather, model, next_date.strftime('%Y-%m-%d'), predictors)
        predictions.append((next_date.strftime('%Y-%m-%d'), round(prediction, 2) if prediction else "No data"))
    return predictions

# Function to reset global variables
def reset_data():
    global weather, predictors, model
    weather = load_weather_data()
    rolling_horizons = [3, 14]
    for horizon in rolling_horizons:
        for col in ['tmax', 'tmin', 'prcp']:
            weather = compute_rolling(weather, horizon, col)
    weather = weather.iloc[14:, :]
    weather = weather.fillna(0)
    for col in ["tmax", "tmin", "prcp"]:
        weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
        weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean)
    predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]
    model = Ridge(alpha=0.1)
    model.fit(weather[predictors], weather['target'])

# Initial data load
reset_data()

# Function to plot the temperatures of the input date and the next 7 days
def plot_temperature(date_str, predictions):
    temps = [predictions[0][1]] + [prediction[1] for prediction in predictions]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temps, marker='o', linestyle='-', color='b', label="Temperature (°F)")
    
    ax.set_title(f"Temperature Forecast ")
    ax.set_xlabel("Temp for 8 Total Days")
    ax.set_ylabel("Temperature (°F)")
    ax.set_ylim(10, 100)
    ax.grid(True)
    ax.legend()

    plot_filename = f"static/temp_plot_{date_str}.png"
    fig.savefig(plot_filename)
    plt.close(fig)
    return plot_filename

# Flask route for the homepage
@app.route('/')
def home():
    reset_data()  
    return render_template('home.html')

# Flask route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    date_str = request.form['date']
    prediction = predict_weather_on_date(weather, model, date_str, predictors)

    if isinstance(prediction, str) and prediction.startswith("Error"):
        return render_template('result.html', 
                               date=date_str, 
                               prediction=prediction, 
                               next_seven_days=[], 
                               plot_img=None)

    next_seven_days = predict_next_seven_days(weather, model, date_str, predictors)

    plot_filename = plot_temperature(date_str, next_seven_days)

    if prediction is not None:
        return render_template('result.html', 
                               date=date_str, 
                               prediction=round(prediction, 2), 
                               next_seven_days=next_seven_days, 
                               plot_img=plot_filename)
    else:
        return render_template('result.html', 
                               date=date_str, 
                               prediction="No prediction available.", 
                               next_seven_days=next_seven_days, 
                               plot_img=plot_filename)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, threaded=False)