import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

plt.rcParams.update({'font.size': 16})  # Increase font size globally

# Given data
sdist = np.array([500, 400, 300, 250, 200, 100, 50, 25])
mergers = np.array([1,   1,   2,   3,   3,   3,  4, 10])

# Define the inverse model function
def inverse_model(x, a):
    return a / (x-6)

# Fit the inverse model to the data using curve_fit
params, covariance = curve_fit(inverse_model, sdist, mergers, p0=(1))

# Extract the optimized parameter 'a'
a = params
print(f"Optimized parameter: a = {a}")

# Calculate fitted values
y_fit = inverse_model(sdist, a)

# Calculate residuals
residuals = mergers - y_fit

# Calculate chi-squared
chi_squared = np.sum(((mergers - y_fit) ** 2) / y_fit)

# Degrees of freedom (DOF) = number of data points - number of parameters
ndof = len(mergers) - len(params)

# Reduced chi-squared
chi_squared_red = chi_squared / ndof

# Calculate p-value
p_value = 1 - chi2.cdf(chi_squared, ndof)

# Print statistics
print(f"Chi-squared: {chi_squared}")
print(f"Reduced Chi-squared: {chi_squared_red}")
print(f"P-value: {p_value}")

# Generate points to plot the fitted curve
x_fit = np.linspace(min(sdist), max(sdist), 100)
y_fit_curve = inverse_model(x_fit, a)

# Create figure with two subplots
fig, ax = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})

# First subplot: Data and fitted curve
ax[0].scatter(sdist, mergers, color='red', label='Data')
ax[0].plot(x_fit, y_fit_curve, label='Fitted Curve', color='blue')
ax[0].set_ylabel('Number of Mergers')
ax[0].legend()
ax[0].set_title('Sticking distance and merger relationship')

# Second subplot: Residuals
ax[1].scatter(sdist, residuals, color='black', label='Residuals')
ax[1].axhline(0, color='gray', linestyle='--')  # Horizontal line at 0
ax[1].set_xlabel('Sticking Distance (AU)')
ax[1].set_ylabel('Residuals')
ax[1].legend()

plt.tight_layout()
plt.show()


# If p-value ≤ α (e.g., p ≤ 0.05): There is enough evidence to reject the null hypothesis. 
#   This means the observed result is statistically significant, and it's unlikely to have occurred by chance if the null hypothesis were true. 

# If p-value > α (e.g., p > 0.05): There is not enough evidence to reject the null hypothesis. 
#   This means the observed result is not statistically significant, and it could have occurred by chance if the null hypothesis were true

# Just say red chi = 1.47 but p = 0.183 so not enough evidence at the 5% level to suggest curve is statistically significant. Need more data points but mergers also have to be integer values.