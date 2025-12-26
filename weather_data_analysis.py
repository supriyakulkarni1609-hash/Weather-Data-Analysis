"""
Weather Data Analysis – Internship Task 1

Author: Supriya Kulkarni
Description:
This script performs weather data analysis including
exploratory data analysis, visualization, and rainfall
prediction using Linear Regression.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def load_data(filepath):
    """Load weather dataset from CSV file."""
    return pd.read_csv(filepath)


def explore_data(df):
    """Perform basic data exploration."""
    print("\nFirst 5 rows:\n", df.head())
    print("\nDataset Info:\n")
    print(df.info())
    print("\nStatistical Summary:\n", df.describe())


def visualize_relationships(df):
    """Visualize relationships between temperature and rainfall."""
    sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
    plt.show()


def monthly_temperature_analysis(df):
    """Analyze monthly average maximum temperature."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month

    monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
    plt.xlabel('Month')
    plt.ylabel('Average Max Temperature')
    plt.title('Monthly Average Max Temperature')
    plt.grid(True)
    plt.show()

    return monthly_avg_max_temp


def rainfall_prediction(df):
    """Predict rainfall using Linear Regression."""
    X = df[['MinTemp', 'MaxTemp']]
    y = df['Rainfall']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\nMean Squared Error for Rainfall Prediction: {mse:.2f}")


def rainfall_insights(df):
    """Identify highest and lowest rainfall months."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month

    monthly_rainfall = df.groupby('Month')['Rainfall'].mean()

    highest_rainfall_month = monthly_rainfall.idxmax()
    lowest_rainfall_month = monthly_rainfall.idxmin()

    print(f"Highest average rainfall month: {highest_rainfall_month}")
    print(f"Lowest average rainfall month: {lowest_rainfall_month}")


def main():
    """Main function to execute analysis."""
    df = load_data('weather.csv')

    # Handle missing values
    df.dropna(inplace=True)

    explore_data(df)
    visualize_relationships(df)
    monthly_temperature_analysis(df)
    rainfall_prediction(df)
    rainfall_insights(df)


if __name__ == "__main__":
    main()
