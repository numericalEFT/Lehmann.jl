import numpy as np
import matplotlib.pylab as plt
from scipy import optimize as optfit

data =  np.loadtxt("./basis/sym_10000_1e-8.dlr", skiprows=1)
data_uni =  np.loadtxt("./basis/universal_10000_1e-8.dlr", skiprows=1)

fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 15}
#fig.set_figheight()
#fig.set_figwidth(6)
plt.rc('font', **font)
#axes[0].set_xlim(0.0, 3.1)
#axes[0].set_ylim(0.06, 0.28)
axes.set_xlabel(r"$\omega$")
axes.set_yticklabels([])
axes.set_yticks([])
value1 = np.zeros(len(data[:,1]))
value2 = np.zeros(len(data_uni[:,1]))
print(data[:,0],data[:,3])
axes.plot(data[:, 1]/max(np.fabs(data[:,1])), value1+1.0, "v", markerfacecolor='none', label="")
axes.plot(data_uni[:, 1]/max(np.fabs(data_uni[:,1])), value2-1.0, "s", markerfacecolor='none', label="")
#axes.plot(data[:, 3]/max(np.fabs(data[:,3])), value1, "*", markerfacecolor='none', label="")

#plt.legend(fontsize = 12,loc="best")
axes.legend(loc="upper left", fontsize = 13, frameon = False) #bbox_to_anchor=(0.30,0.800), fontsize = 5)
plt.savefig("sym_grid.pdf")
