import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem, t

# Dataset of population every year
h_data = np.array([30,47.2,70.2,77.4,36.3,20.6,18.1,21.4,22,25.4,27.1,40.3,57,76.6,52.3,19.5,11.2,7.6,14.6,16.2,24.7]) 
l_data = np.array([4,6.1,9.8,35.2,59.4,41.7,19,13,8.3,9.1,7.4,8,12.33,19.5,45.7,51.1,29.7,15.8,9.7,10.1,8.6]) 
x_data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])


def sine_model(x, A, B, C, D):
    return A * np.sin(B * x + C) + D
initial_guess_h = [np.ptp(h_data)/2, 2 * np.pi / len(x_data), 0, np.mean(h_data)]
initial_guess_l = [np.ptp(l_data)/2, 2 * np.pi / len(x_data), 0, np.mean(l_data)]
params_h, _ = curve_fit(sine_model, x_data, h_data, p0=initial_guess_h)
params_l, _ = curve_fit(sine_model, x_data, l_data, p0=initial_guess_l)


x_fit = np.linspace(min(x_data), max(x_data), 1000)
y_fit_h = sine_model(x_fit, *params_h)
y_fit_l = sine_model(x_fit, *params_l)


# Calculate standard error of the mean (SEM) and the confidence intervals
h_se = sem(h_data)
l_se = sem(l_data)
confidence = 0.95
h_ci = h_se * t.ppf((1 + confidence) / 2., len(h_data)-1)
l_ci = l_se * t.ppf((1 + confidence) / 2., len(l_data)-1)


# The original data points
plt.scatter(x_data, h_data, label='Hares Data Points', color='green')
plt.scatter(x_data, l_data, label='Lynx Data Points', color='red')

# The fitted sine curves
plt.plot(x_fit, y_fit_h, color='green', label='Fitted Sine Curve for Hares')
plt.plot(x_fit, y_fit_l, color='red', label='Fitted Sine Curve for Lynx')

# Confidence interval 
plt.fill_between(x_fit, y_fit_h - h_ci, y_fit_h + h_ci, color='green', alpha=0.2, label='95% CI for Hares')
plt.fill_between(x_fit, y_fit_l - l_ci, y_fit_l + l_ci, color='red', alpha=0.2, label='95% CI for Lynx')


# Mean lines 
plt.axhline(np.mean(h_data), color='green', linestyle='dashed', label='Mean for Hares')
plt.axhline(np.mean(l_data), color='red', linestyle='dashed', label='Mean for Lynx')


plt.xlabel('Time Steps')
plt.ylabel('Population x1000')
plt.title('Sine Curve Fit to Hare and Lynx Data with Confidence Intervals')
plt.legend()
plt.show()

# Print the fitted parameters for both datasets
print("Fitted parameters for H data:", params_h)
print("Fitted parameters for L data:", params_l)




