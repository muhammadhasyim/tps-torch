import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(5,5))


#Plot the Quartic potential
def V(x):
    return (1-x**2)**2
x = np.linspace(-1.6,1.6)
ax.plot(x,V(x),'k-')

#Loading the string and brownian particle trajectory here
num_nodes = 6
strings = []
bp = []
for i in range(num_nodes):
    strings.append(np.loadtxt("string_{}.xyz".format(i+1)))
    bp.append(np.loadtxt("test_bp_{}.txt".format(i+1)))

#Set the initial plotting 
string_plots = ()
bp_plots = ()
for j in range(num_nodes):
    line = ax.axvline(0)
    string_plots += (line,)
    line, = ax.plot([],[],'bo',markersize=5)
    bp_plots += (line,)

def animate(i):
    for j in range(num_nodes):
        string_plots[j].set_xdata(strings[j][i])#+strings[j-1][i]))
        bp_plots[j].set_data(bp[j][i],V(bp[j][i]))
    return string_plots+bp_plots

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig, animate, min(len(strings[0]),len(bp[0])), interval=1, repeat=False,blit=True)

#Uncomment for saving
#ani.save('im.mp4', writer=writer)

plt.show()
