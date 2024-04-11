# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE

import matplotlib.pyplot as plt
import numpy as np

def f1(x) :
    return 0.5*(np.e**x+ np.e**-x)
def f2(t) :
    return np.cos(t)*(1 - np.cos(t)), np.sin(t)*(1 - np.cos(t))
def f3(x, y) :
    return np.sin(np.sqrt(x**2 + 2 * y**2))


#input/output for a)
x_a = np.linspace(-2, 2, 100)
y_a = f1(x_a)
#input/output for b)
t_b = np.linspace(0, 2*np.pi, 100)
x_b, y_b = f2(t_b)
# Create a figure with two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10,4))

# Plot data on the first subplot
ax1.plot(x_a, y_a, color='blue', label='a)')
ax1.set_title('a)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()

# Plot data on the second subplot
ax2.plot(x_b, y_b, color='red', label='b)')
ax2.set_title('b)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()


# Adjust layout and display the plot
# plt.tight_layout()
plt.show()

#input/output for b)
x_c = np.linspace(-2*np.pi, 2*np.pi, 100)
y_c = np.linspace(-2*np.pi, 2*np.pi, 100) 
x, y = np.meshgrid(x_c, y_c)
f3_c = f3(x, y)

fig = plt.figure(figsize=(8, 6))

# Add a 3D axis to the figure
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(x, y, f3_c, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('c)')

plt.show()

#d
def f4(t) :
    ret1 = 2 * np.cos(t) - 1, 2 * np.sin(t)
    ret2 = 2 * np.cos(t) + 1, 2 * np.sin(t)
    return ret1, ret2

t_d = np.linspace(0, 2 * np.pi, 100)
(x1, y1), (x2, y2) = f4(t_d)
plt.plot(x1, y1, x2, y2, color='red')
plt.fill_between(x1, y1, color='red')
plt.fill_between(x2, y2, color='red')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Fill Plot')

# Add legend
plt.legend()

# Display the plot
plt.show()
