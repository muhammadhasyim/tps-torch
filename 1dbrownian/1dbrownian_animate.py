import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

#for i in range(epochs*len(X_train)):
fig, ax = plt.subplots(figsize=(5,5))
def V(x):
    return (1-x**2)**2

#I'm going to load the string and brownian particle trajectory here
num_nodes = 12+2#8
strings = []
bp = []
for i in range(num_nodes-2):
    strings.append(np.loadtxt("string_{}.txt".format(i+1)))
    bp.append(np.loadtxt("bp_{}.txt".format(i+1)))
x = np.linspace(-1.6,1.6)

ax.plot(x,V(x),'k-')
string_plots = ()
bp_plots = ()
for j in range(num_nodes-1):
    line = ax.axvline(0)#0.5*(strings[j][i]+strings[j-1][i]))
    string_plots += (line,)
    if j < num_nodes-2:
        line, = ax.plot([],[],'bo',markersize=5)#bp[j-1][i],V(bp[j-1][i]))
        bp_plots += (line,)
def animate(i):
    #if i in index:
    for j in range(num_nodes-1):
        if j == 0:
            string_plots[j].set_xdata(0.5*(-1.0+strings[j][i]))
            bp_plots[j].set_data(bp[j][i],V(bp[j][i]))
        elif j ==  num_nodes-2:
            string_plots[j].set_xdata(0.5*(1.0+strings[j-1][i]))
        elif j < num_nodes-2:
            string_plots[j].set_xdata(0.5*(strings[j][i]+strings[j-1][i]))
            bp_plots[j].set_data(bp[j][i],V(bp[j][i]))
    return string_plots+bp_plots
# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig, animate, len(strings[0]), interval=200, repeat=False,blit=True)
#ani.save('im.mp4', writer=writer)
plt.show()
