import matplotlib.pyplot as plt
import numpy as np
file = open('utils/plots/cnv_epoch', 'r')
cnv_epochs = []
for line in file.readlines():
    cnv_epochs.append(float(line.split(" ")[1]))

  
print(np.mean(cnv_epochs))
print(np.std(cnv_epochs))

"""# Using readlines()
file = open('utils/plots/l1_train', 'r')
train_loss = []
for line in file.readlines():
    train_loss.append(float(line.split(" ")[1]))

# Using readlines()
file = open('utils/plots/l1_val', 'r')
val_loss = []
for line in file.readlines():
    val_loss.append(float(line.split(" ")[1]))



epoch = list(range(len(train_loss)))

plt.figure()
  
# Plotting both the curves simultaneously
plt.plot(epoch, train_loss, label='Train loss')
plt.plot(epoch, val_loss, label='Validation loss')
  
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VQ-VAE2 Overfit training")
plt.grid(True)
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()"""

