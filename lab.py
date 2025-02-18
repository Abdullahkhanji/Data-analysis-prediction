import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_and_filter(filename, filter_limit):
    df = pd.read_csv(filename)
    df["Y"] = pd.to_numeric(df["Y"])
    
    filtered_df = df[df["Y"] <= filter_limit]
    filtered_df.to_csv("filtered_data.csv", index=False)
    return filtered_df

def fix_deformation(dataframe):

    data = dataframe.to_numpy()
    
    data[:, 1] *= 2
    
    data[:, 1] -= 5
    
    theta = 45 * np.pi / 180 
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_new = data[:, 0] * cos_theta - data[:, 1] * sin_theta
    y_new = data[:, 0] * sin_theta + data[:, 1] * cos_theta
    data[:, 0], data[:, 1] = x_new, y_new
    
    data[:, 0] /= 0.01
    
    data[:, 0] += 1500
    
    data[:, 0] = np.round(data[:, 0]).astype(int)


    np.savetxt("filtered_data.csv", data, delimiter=',')
    return data

def fit_and_predict(dataset, day):
    x_values = dataset[:, 0]
    y_values = dataset[:, 1]
    
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    
    beta = np.sum((x_values - x_mean) * (y_values - y_mean)) / np.sum((x_values - x_mean) ** 2)
    
    alpha = y_mean - beta * x_mean
    
    predicted_y = alpha + beta * day
    return (alpha, beta, predicted_y)

def plot(dataset, alpha, beta, prediction, day):
    x_values = dataset[:, 0]
    y_values = dataset[:, 1]

    sorted_indices = np.argsort(x_values)
    x_values = x_values[sorted_indices]
    y_values = y_values[sorted_indices]

    plt.figure(figsize=(8, 6))


    plt.scatter(x_values, y_values, color= "red")

    x_line = np.linspace(min(x_values), day, 100)
    y_line = alpha + beta * x_line
    plt.plot(x_line, y_line, color="green")


    predicted_y = alpha + beta * day
    plt.scatter(day, predicted_y, color="blue", marker="o", label=f"Prediction for x={day}")
    plt.xlabel("Day")
    plt.ylabel("USD")
    plt.title("Exchange Rate")

    plt.show()

def test():
    filtered_data = read_and_filter('data.csv', 11)
    fixed_data = fix_deformation(filtered_data)
    alpha, beta, prediction = fit_and_predict(fixed_data, 2700)
    plot(fixed_data, alpha, beta, prediction, 2700)

test()