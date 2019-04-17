import os
import matplotlib.pyplot as plt
l = os.listdir("./loss_write/")
d_loss_list = []
g_loss_list = []
for i in range(0, len(l), 10):
    f = open("./loss_write/dg_loss%d.log"%(i), "r")
    line = f.readline()
    splited_line = line.split()
    d_loss = float(splited_line[0])
    g_loss = float(splited_line[1])
    d_loss_list.append(d_loss)
    g_loss_list.append(g_loss)
"""
plt.title("Generator")
plt.plot(g_loss_list)
"""
plt.title("Discriminator")
plt.plot(d_loss_list)
plt.show()
