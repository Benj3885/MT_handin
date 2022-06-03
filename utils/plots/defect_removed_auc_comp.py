import matplotlib.pyplot as plt

fixed_auc = [0.659, 0.815, 0.853, 0.911, 0.943, 0.951]
random_auc = [0.629, 0.850, 0.923, 0.943, 0.947, 0.950]

fixed_std = [0.0726, 0.0762, 0.0955, 0.0579, 0.0126, 0.00961]
random_std = [0.0855, 0.0430, 0.0188, 0.0157, 0.0106, 0.00886]

P = [0, 0.2, 0.4, 0.6, 0.8, 1]

plt.figure(1)
plt.subplot(211)
  
# Plotting both the curves simultaneously
plt.plot(P, fixed_auc, label='Grouped pairs')
plt.plot(P, random_auc, label='Non-grouped pairs')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("P")
plt.ylabel("Mean AUC")
plt.title("Grouped and non-grouped images comparison")
plt.grid(True)
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

plt.subplot(212)

# Plotting both the curves simultaneously
plt.plot(P, fixed_std, label='Grouped pairs')
plt.plot(P, random_std, label='Non-grouped pairs')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("P")
plt.ylabel("Std. of AUC")
plt.grid(True)
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()

