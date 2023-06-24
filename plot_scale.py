import numpy as np
import matplotlib.pylab as plt
from scipy import optimize as optfit
plt.style.use(['science'])
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 15}
#fig.set_figheight()
#fig.set_figwidth(6)
plt.rc('font', **font)

data =  np.loadtxt("./basis/sym_10000_1e-8.dlr", skiprows=1)
data_uni =  np.loadtxt("./basis/universal_10000_1e-8.dlr", skiprows=1)
Lambda = 10000.0
fig, (ax1, ax2,ax3) = plt.subplots(3, 1,  figsize=(10, 8))
#fig, ax1 = plt.subplots(nrows=1, ncols=1, sharey=True)
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 15}
#fig.set_figheight()
#fig.set_figwidth(6)
plt.rc('font', **font)
#ax1[0].set_xlim(0.0, 3.1)
#ax1[0].set_ylim(0.06, 0.28)
value1 = np.zeros(len(data[:,1]))
value2 = np.zeros(len(data_uni[:,1]))
shift = 0.6


x_start = -0.13
x_end = -0.37
y_start = -1.3*shift
y_end = 1.3*shift

# Create the box with fixed size
ax1.add_patch(plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                          facecolor= 'white',edgecolor='green', linewidth = 2.0,alpha=0.5))
x_start = -x_start
x_end = -x_end
ax1.add_patch(plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                          facecolor= 'white',edgecolor='green', linewidth = 2.0,alpha=0.5))
ax1.set_xlabel(r"$\omega/\Lambda $")
ax1.set_yticklabels([])
ax1.set_yticks([])
ax1.set_ylim(-shift*1.5,shift*1.5)
#print(data[:,0],data[:,3])
ax1.plot(data[:, 1]/Lambda, value1+shift, "v", markerfacecolor='none', label="SDLR")
ax1.plot(data_uni[:, 1]/Lambda, value2-shift, "s", markerfacecolor='none', label="DLR")
#ax2.plot(data[:, 3]/max(np.fabs(data[:,3])), value1, "*", markerfacecolor='none', label="")
#plt.legend(fontsize = 12,loc="best")
ax1.legend(loc="right", fontsize = 16, frameon = False) #bbox_to_anchor=(0.30,0.800), fontsize = 5)




ax2.set_xlabel(r"$\tau/\beta$")
ax2.set_yticklabels([])
ax2.set_yticks([])
ax2.set_ylim(-shift*1.5,shift*1.5)


x_start = 0.16
x_end = 0.24
y_start = -1.3*shift
y_end = 1.3*shift

# Create the box with fixed size
ax2.add_patch(plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                          facecolor= 'white',edgecolor='green', linewidth = 2.0,alpha=0.5))
x_start = 1.0 -x_start
x_end = 1.0 -x_end
ax2.add_patch(plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                          facecolor= 'white',edgecolor='green', linewidth = 2.0,alpha=0.5))


#print(data[:,0],data[:,3])
ax2.plot(data[:, 2], value1+shift, "v", markerfacecolor='none', label="SDLR")
ax2.plot(data_uni[:, 2], value2-shift, "s", markerfacecolor='none', label="DLR")
#ax2.plot(data[:, 3]/max(np.fabs(data[:,3])), value1, "*", markerfacecolor='none', label="")
#plt.legend(fontsize = 12,loc="best")
#ax2.legend(loc="right", fontsize = 13, frameon = False) #bbox_to_anchor=(0.30,0.800), fontsize = 5)


x_start = -3
x_end = -5.5
y_start = -1.3*shift
y_end = 1.3*shift
ax3.add_patch(plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                          facecolor= 'white',edgecolor='green', linewidth = 2.0,alpha=0.5))
x_start =  -x_start
x_end =  -x_end
ax3.add_patch(plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                          facecolor= 'white',edgecolor='green', linewidth = 2.0,alpha=0.5))


x_start = -0.4
x_end = -1.3
y_start = -1.3*shift
y_end = 1.3*shift
ax3.add_patch(plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                          facecolor= 'white',edgecolor='green', linewidth = 2.0,alpha=0.5))
x_start =  -x_start
x_end =  -x_end
ax3.add_patch(plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                          facecolor= 'white',edgecolor='green', linewidth = 2.0,alpha=0.5))


ax3.set_xlabel(r"$\omega_n/\Lambda$")
ax3.set_yticklabels([])
ax3.set_yticks([])
ax3.set_ylim(-shift*1.5,shift*1.5)
#print(data[:,0],data[:,3])
data[:,3] = (data[:,3]*2+1)*np.pi
data_uni[:, 3] = (data_uni[:,3]*2+1)*np.pi
ax3.set_xlim(-1.1*data_uni[-1, 3]/Lambda,1.1*data_uni[-1, 3]/Lambda)
ax3.plot(data[:, 3]/Lambda, value1+shift, "v", markerfacecolor='none', label="SDLR")
ax3.plot(data_uni[:, 3]/Lambda, value2-shift, "s", markerfacecolor='none', label="DLR")
#ax2.plot(data[:, 3]/max(np.fabs(data[:,3])), value1, "*", markerfacecolor='none', label="")
fig.subplots_adjust(hspace=0.3)
plt.tight_layout()
#plt.legend(fontsize = 12,loc="best")
#ax3.legend(loc="right", fontsize = 13, frameon = False) #bbox_to_anchor=(0.30,0.800), fontsize = 5)
plt.savefig("sym_grid.pdf")
