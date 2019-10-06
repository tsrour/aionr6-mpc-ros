import matplotlib
import matplotlib.pyplot as plt
import os

# plot the results
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.scatter(histX,histY)
ax1.set_title('XY Plot')
ax2 = fig.add_subplot(222)
ax2.scatter(times,implementedU1)
ax2.set_title('Angular Velocity (W)')
ax3 = fig.add_subplot(223)
ax3.scatter(times,implementedU2)
ax3.set_title('Linear Acceleration (a)')
ax4 = fig.add_subplot(224)
ax4.scatter(times,implementedV)
ax4.set_title('Linear Velocity (v)')
fig.savefig('output.png')
os.system("xdg-open output.png")
