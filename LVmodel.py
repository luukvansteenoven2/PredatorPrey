import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#The Lotka-Volterra equations
def eqs(x, t, alpha, beta, gamma, delta):
    return [alpha * x[0] - beta * x[0] * x[1],  # Prey growth equation
            delta * x[0] * x[1] - gamma * x[1]]  # Predator growth equation

# Parameters
alpha = 0.01   # Prey growth rate
beta = 0.0005  # Predation rate
gamma = 0.5   # Predator mortality rate
delta = 0.0002  # Predator growth rate by consuming prey


# Initial population
y0 = [2000, 5]
# Time grid
t = np.linspace(0, 1000, 1000)
# ODE system with the updated parameters
sol = odeint(eqs, y0, t, args=(alpha, beta, gamma, delta))
# Calculate mean values
prey_mean = np.mean(sol[:, 0])
predator_mean = np.mean(sol[:, 1])

fig, ax1 = plt.subplots()

color_prey = 'blue'
ax1.set_xlabel('Time')
ax1.set_ylabel('Hares Population', color=color_prey)
ax1.plot(t, sol[:, 0], label='Hares', color=color_prey, linestyle='-')
ax1.tick_params(axis='y', labelcolor=color_prey)
ax1.axhline(prey_mean, color=color_prey, linestyle='--')  
ax1.text(0, prey_mean, f' {prey_mean:.2f}', va='center', ha='right', color=color_prey, fontsize=10, backgroundcolor='white')

ax2 = ax1.twinx()
color_predator = 'red'
ax2.set_ylabel('Lynx Population', color=color_predator)
ax2.plot(t, sol[:, 1], label='Lynx', color=color_predator, linestyle='-')
ax2.tick_params(axis='y', labelcolor=color_predator)
ax2.axhline(predator_mean, color=color_predator, linestyle='--') 
ax2.text(1000, predator_mean, f' {predator_mean:.2f}', va='center', ha='left', color=color_predator, fontsize=10, backgroundcolor='white')



plt.tight_layout()
plt.show()
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))

plt.figure()
H = sol[:, 0]
L = sol[:, 1]
plt.plot(H, L)
plt.title('Hare/Lynx Phase Plot')
plt.xlabel('Hare Population')
plt.ylabel('Lynx Population')
plt.grid(True)

plt.tight_layout()
plt.show()

