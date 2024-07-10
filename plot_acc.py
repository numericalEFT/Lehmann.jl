import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
plt.style.use(['science'])
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 14}
#fig.set_figheight()
#fig.set_figwidth(6)
plt.rc('font', **font)

data1 =  np.loadtxt("./test/accuracy_test1.dat")
data2 =  np.loadtxt("./test/accuracy_test2.dat")
data3 = np.loadtxt("./test/accuracy_test3.dat")

x1 =np.log10(data1[:,0])
#sdlr1 = data2[:,-1]
#dlr1 = data1[:,-1]

# Creating subplots with shared y-axis
#fig, (ax1) = plt.subplots()
fig, (ax1,ax2) = plt.subplots(1, 2,  figsize=(8, 4))
# Plotting data1
ax1.plot(x1, data1[:,1],  'o-' ,  label=r'$L2$')
ax1.plot(x1, data1[:,2],  '^--',  label=r'$L_{\inf}$')
ax1.set_xlabel(r'$\epsilon$')
ax1.set_ylabel(r'$\eta/\epsilon$')
#ax1.set_ylim([-2.0, 2.0])
ax1.set_yscale('log')
# ax1.set_title('Data 1')
#ax1.set_yscale('log')
ax1.set_xticks(x1)
ax1.set_xticklabels([f'$10^{{{int(x)}}}$' for x in x1])
#ylist = [-2.0,-1.0,0.0,1.0,2.0]
#ax1.set_yticks(ylist, [f'$10^{{{int(x)}}}$' for x in ylist])
ax1.legend(loc = 'upper left')

ax2.plot(x1, data2[:,1],  'o-' ,  label=r'$L2$')
ax2.plot(x1, data2[:,2],  '^--',  label=r'$L_{\inf}$')
ax2.set_xlabel(r'$\epsilon$')
ax2.set_ylabel(r'$\eta/\epsilon$')
#ax2.set_ylim([-2.0, 2.0])
#ax2.set_yscale('log')
# ax2.set_title('Data 1')
#ax2.set_yscale('log')
ax2.set_xticks(x1)
ax2.set_xticklabels([f'$10^{{{int(x)}}}$' for x in x1])
#ylist = [-2.0,-1.0,0.0,1.0,2.0]
#ax2.set_yticks(ylist, [f'$10^{{{int(x)}}}$' for x in ylist])
ax2.legend(loc = 'upper left')
plt.tight_layout()
plt.savefig("acc_sampling.pdf")


fig, (ax3) = plt.subplots()

ax3.plot(x1, data3[:,1],  'o-' ,  label=r'$L2$')
ax3.plot(x1, data3[:,2],  '^--',  label=r'$L_{\inf}$')
ax3.set_xlabel(r'$\epsilon$')
ax3.set_ylabel(r'$\eta/\epsilon$')
#ax3.set_ylim([-2.0, 2.0])
ax3.set_yscale('log')
# ax3.set_title('Data 1')
#ax3.set_yscale('log')
ax3.set_xticks(x1)
ax3.set_xticklabels([f'$10^{{{int(x)}}}$' for x in x1])
ax3.set_ylim(0.1, 1000)
#ylist = [-2.0,-1.0,0.0,1.0,2.0]
#ax3.set_yticks(ylist, [f'$10^{{{int(x)}}}$' for x in ylist])
ax3.legend()
plt.tight_layout()
plt.savefig("acc_L2.pdf")

