import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import make_interp_spline
plt.style.use(['science'])
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 17}
#fig.set_figheight()
#fig.set_figwidth(6)
plt.rc('font', **font)
data3 = np.loadtxt("./src/vertex3/res_matsu.dat")
data2 = np.loadtxt("./src/vertex3/res_tau.dat")
data1 = np.loadtxt("./src/vertex3/res_freq.dat")


# Creating subplots with shared y-axis
fig, (ax1, ax2,ax3) = plt.subplots(1, 3,  figsize=(10, 5))

# Plotting data1
#ax1.plot(data1[:,0],data1[:,1]*data1[:,0]**2 , 'o-' ,  label='SDLR')
def smooth_curve( x , y, N):
        x_smooth = np.linspace(x.min(), x.max(), N)  # Generate more points for smoothness
        spl = make_interp_spline(x, y)
        y_smooth = spl(x_smooth)
        x_smooth = np.concatenate((x_smooth,x))
        y_smooth = np.concatenate((y_smooth,y))
        sorted_indices = np.argsort(x_smooth)
        return x_smooth[sorted_indices] , y_smooth[sorted_indices]

x_sym = np.concatenate((-np.flip(data1[:,0]), data1[:,0]) )
y_sym = np.concatenate((np.flip(data1[:,1]), data1[:,1] ))
#x1, y1 = x_sym,y_sym
x1,y1 = smooth_curve(x_sym,y_sym,200)
ax1.set_ylim(0,1.4e-15)
ax1.plot(x1,y1, '-' )
ax1.set_xlabel(r'$\omega$')
ax1.set_ylabel('residual')
# ax1.set_title('Data 1')
#ax1.set_yscale('log')
#ax1.set_xticklabels([f'$10^{{-{x}}}$' for x in x1])
# ax1.legend()

# # Plotting data2
#x2,y2 = data2[:,0],data2[:,1]
x2,y2 = smooth_curve(data2[:,0],data2[:,1],200)
ax2.plot(x2,y2, '-' )
ax2.set_ylim(0,1e-11)
ax2.set_xlabel(r'$\tau$')
ax2.set_ylabel('residual')
# ax1.set_title('Data 1')
#ax1.set_yscale('log')
#ax1.set_xticklabels([f'$10^{{-{x}}}$' for x in x1])
# ax2.legend()

# # Plotting data2
data3[:,0]  = (data3[:,0] *2 +1)* np.pi 
middle = len(data3[:,0])//2

inset_ax = ax3.inset_axes([0.2, 0.7, 0.6, 0.2]) 
cut2 = 600
cut3 =640

inset_ax.plot( data3[cut2:cut3,0],data3[cut2:cut3,1], linestyle = "dashed", linewidth = 4,  label = r"residual") 
inset_ax.plot( data3[cut2:cut3,0], data3[cut2,1]*data3[cut2,0]**2 / data3[cut2:cut3,0] **2, label = r"$\propto\omega_n^{-2}$" ) 
inset_ax.legend(fontsize=9.5, facecolor='white', framealpha=1.0,frameon = False)#, bbox_to_anchor = [0.39,0.39])
cut = 62
#ax3.plot(data3[middle-cut:middle+cut,0],data3[middle-cut:middle+cut,1]*data3[middle-cut:middle+cut,0]**2, '-' )
#x3, y3 = data3[middle-cut:middle+cut,0],data3[middle-cut:middle+cut,1]
x3,y3= smooth_curve(data3[middle-cut:middle+cut,0] , data3[middle-cut:middle+cut,1], 50)
ax3.set_ylim(0, 1.5e-16)
ax3.plot(x3,y3, '-' )
ax3.set_xlabel(r'$\omega_n$')
ax3.set_ylabel('residual')
# ax1.set_title('Data 1')
#ax1.set_yscale('log')
#ax1.set_xticklabels([f'$10^{{-{x}}}$' for x in x1])
# ax3.legend()

# formatter = ticker.FuncFormatter(lambda x,pos: '{:.0f}'.format(x))

# ax2.yaxis.set_major_formatter(formatter)
# # Adjusting spacing between subplots
plt.tight_layout()
plt.savefig("sc.pdf")
# Displaying the plot
plt.show()
