# Least Squares Regression on Tips Dataset:

This project performs a linear regression analysis on the [Tips dataset](https://github.com/mwaskom/seaborn-data/blob/master/tips.csv) using various least squares methods. The goal is to model the relationship between the total bill and the tip amount given at a restaurant.

---

## 📁 Project Structure

``` 
least_square_project/
├── tips.csv # Input dataset (CSV format)
├── project.py # Main script with data loading, regression, and plotting
├── README.md # This file

```

---

## 🔧 Methods Used

This project solves the overdetermined system \( Ax = b \) using the following techniques:

- **Normal Equation**:
  \( x = (A^T A)^{-1} A^T b \)

- **`scipy.linalg.lstsq()`**:
  More numerically stable; uses QR decomposition or SVD under the hood.

- **`scipy.linalg.pinv()`**:
  Uses the Moore-Penrose pseudo-inverse computed via SVD.

Each method estimates the best-fit line between total bill and tip amount.

---

## 📊 Output

The program prints:
- Intercept and slope of the best-fit line
- Comparison of results from `lstsq` and `pinv`
- A plot of the original data points and the fitted regression line

---

## ▶️ How to Run

1. Install required packages:
   ```
   bash
   pip install pandas numpy matplotlib scipy

   ```
