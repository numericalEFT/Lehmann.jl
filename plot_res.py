import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
plt.style.use(['science'])
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 15}
#fig.set_figheight()
#fig.set_figwidth(6)
plt.rc('font', **font)
data1 = [[6, 52, 45],
         [8, 66, 57],
         [10, 80, 70],
         [12, 92, 83],
         [14, 102, 95]]

data2 = [[3, 54, 51],
         [4, 80, 70],
         [5, 98, 90],
         [6, 128, 110],
         [7, 154, 128]]

# Extracting columns for data1
x1 = [row[0] for row in data1]
sdlr1 = [row[1] for row in data1]
dlr1 = [row[2] for row in data1]

# Extracting columns for data2
x2 = [row[0] for row in data2]
sdlr2 = [row[1] for row in data2]
dlr2 = [row[2] for row in data2]

# Creating subplots with shared y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))


# Apply the formatter to the x-axis

# Plotting data1
ax1.plot(x1, sdlr1, 'o-' ,  label='SDLR')
ax1.plot(x1, dlr1, '^', linestyle = 'dashed', label='DLR')
ax1.set_xlabel(r'$\epsilon$')
ax1.set_ylabel('N')
# ax1.set_title('Data 1')
#ax1.set_yscale('log')
ax1.set_xticks(x1)
ax1.set_xticklabels([f'$10^{{-{x}}}$' for x in x1])
ax1.legend()

# Plotting data2
ax2.plot(x2, sdlr2,'o-' ,label='SDLR')
ax2.plot(x2, dlr2, '^',  linestyle = "dashed"  ,label='DLR')
ax2.set_xlabel(r'$\Lambda$')
# ax2.set_title('Data 2')
#ax2.set_yscale('log')
ax2.set_xticks(x2)
ax2.set_xticklabels([f'$10^{{{x}}}$' for x in x2])
ax2.legend()
formatter = ticker.FuncFormatter(lambda x,pos: '{:.0f}'.format(x))

ax2.yaxis.set_major_formatter(formatter)
# Adjusting spacing between subplots
plt.tight_layout()
plt.savefig("sc.pdf")
# Displaying the plot
plt.show()
