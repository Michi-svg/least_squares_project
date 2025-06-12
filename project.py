import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(path, sep="\t")  # tab-separated
    return df

def clean_data(df):
    """Clean and prepare the data."""
    df = df.dropna()  # Remove any missing values
    return df

def build_design_matrix(df):
    """Construct matrix A and vector b from cleaned DataFrame."""
    A = np.column_stack((np.ones(len(df)), df['total_bill']))  # Add intercept term
    b = df['tip'].values
    return A, b

def least_squares(A, b):
    """Solve the least squares problem using the normal equation."""
    x_hat = np.linalg.inv(A.T @ A) @ A.T @ b
    return x_hat

def plot_results(df, x_hat):
    """Plot the original data and fitted line."""
    x = df['total_bill']
    y = df['tip']
    y_pred = x_hat[0] + x_hat[1] * x

    plt.scatter(x, y, label='Data')
    plt.plot(x, y_pred, color='red', label='Best fit line')
    plt.xlabel('Total Bill')
    plt.ylabel('Tip')
    plt.legend()
    plt.title('Least Squares Regression: Tip vs Total Bill')
    plt.show()

def main():
    # Step 1: Load and clean data
    df = load_data("tips.csv")
    df = clean_data(df)

    # Step 2: Build matrices
    A, b = build_design_matrix(df)

    # Step 3: Solve least squares
    x_hat = least_squares(A, b)

    # Step 4: Print result
    print(f"Intercept: {x_hat[0]:.4f}")
    print(f"Slope: {x_hat[1]:.4f}")

    # Step 5: Plot results
    plot_results(df, x_hat)

if __name__ == "__main__":
    main()

