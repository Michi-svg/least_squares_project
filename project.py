import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq, pinv

def load_data(path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(path)  # Use default comma separator
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

def least_squares_lstsq(A, b):
    """Solve using scipy.linalg.lstsq()"""
    x_hat, residuals, rank, s = lstsq(A, b)
    return x_hat

def least_squares_pinv(A, b):
    """Solve using pseudo-inverse"""
    x_hat = pinv(A) @ b
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

    # Step 3a: Solve using lstsq
    x_hat_lstsq = least_squares_lstsq(A, b)
    print("Using scipy.linalg.lstsq():")
    print(f"Intercept: {x_hat_lstsq[0]:.4f}")
    print(f"Slope: {x_hat_lstsq[1]:.4f}")
    print()

    # Step 3b: Solve using pseudo-inverse
    x_hat_pinv = least_squares_pinv(A, b)
    print("Using scipy.linalg.pinv():")
    print(f"Intercept: {x_hat_pinv[0]:.4f}")
    print(f"Slope: {x_hat_pinv[1]:.4f}")
    print()

    # Step 4: Plot results (you can use either x_hat_lstsq or x_hat_pinv)
    plot_results(df, x_hat_lstsq)

if __name__ == "__main__":
    main()

