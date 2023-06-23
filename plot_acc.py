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

data =  np.loadtxt("./test/accuracy.dat")
data1 = data[::2,:] 
data2 = data[1::2,:] 
# Extracting columns for data1
x1 =-data1[:,0]
sdlr1 = data2[:,2]
dlr1 = data1[:,2]

# Creating subplots with shared y-axis
fig, (ax1) = plt.subplots()

# Plotting data1
ax1.plot(x1, sdlr1, 'o-' ,  label='SDLR')
ax1.plot(x1, dlr1, '^', linestyle = 'dashed', label='DLR')
ax1.set_xlabel(r'$\epsilon$')
ax1.set_ylabel(r'$\eta$')
ax1.set_yscale('log')
# ax1.set_title('Data 1')
#ax1.set_yscale('log')
ax1.set_xticks(x1)
ax1.set_xticklabels([f'$10^{{{int(x)}}}$' for x in x1])
ax1.legend()

#formatter = ticker.FuncFormatter(lambda x,pos: '{:.0f}'.format(x))

#ax2.yaxis.set_major_formatter(formatter)
# Adjusting spacing between subplots
plt.tight_layout()
plt.savefig("acc.pdf")
# Displaying the plot
plt.show()
