import matplotlib.pyplot as plt

AUCs1 = []

for fn in ['utils/plots/avg_AUC_1', 'utils/plots/avg_AUC_2']:
    file = open(fn, 'r')
    for line in file.readlines():
        AUCs1.append(float(line.split(" ")[1]))

AUCs2 = []
file = open('utils/plots/avg_AUC_old', 'r')
for line in file.readlines():
    AUCs2.append(float(line.split(" ")[1]))

it1 = [x+1 for x in range(len(AUCs1))]
it2 = [x+1 for x in range(len(AUCs2))]

plt.figure()
  
# Plotting both the curves simultaneously
plt.plot(it1, AUCs1, label='Augmented good')
plt.plot(it2, AUCs2, label='Augmented both')


# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Iteration")
plt.ylabel("Mean AUC")
plt.title("Mean AUCs for each iteration in experiments")
plt.grid(True)
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()

