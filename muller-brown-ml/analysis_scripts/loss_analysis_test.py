import numpy as np # numerical array library
import matplotlib.pyplot as plt
import matplotlib as mpl
#plt.switch_backend('agg')
from matplotlib.ticker import AutoMinorLocator
mpl.rcParams['text.usetex'] = True
params= {'text.latex.preamble' : [r'\usepackage{bm}',r'\usepackage{mathtools,amsmath}']}
mpl.rcParams.update(params)

# Do loss analysis
folders = [];
for i in range(0,4):
    for j in range(0,5):
        folders.append(str(i)+"_"+str(j)+"/")

loss_grad = np.zeros((len(folders),20001), dtype=float)
loss_bc = np.zeros((len(folders),20001), dtype=float)
loss_cl = np.zeros((len(folders),20001), dtype=float)
loss_fts_cl = np.zeros((len(folders),20001), dtype=float)
loss_total = np.zeros((len(folders),20001), dtype=float)

for i in range(len(folders)):
    data = np.genfromtxt(folders[i]+"simple_loss.txt")
    print(folders[i])
    loss_grad[i,:] = data[:,1]
    loss_bc[i,:] = data[:,2]

loss_total = loss_grad+loss_bc+loss_cl+loss_fts_cl

loss_grad_avg = np.zeros((4,20001), dtype=float)
loss_grad_std = np.zeros((4,20001), dtype=float)
loss_bc_avg = np.zeros((4,20001), dtype=float)
loss_bc_std = np.zeros((4,20001), dtype=float)
loss_cl_avg = np.zeros((4,20001), dtype=float)
loss_cl_std = np.zeros((4,20001), dtype=float)
loss_fts_cl_avg = np.zeros((4,20001), dtype=float)
loss_fts_cl_std = np.zeros((4,20001), dtype=float)
loss_total_avg = np.zeros((4,20001), dtype=float)
loss_total_std = np.zeros((4,20001), dtype=float)
for i in range(0,4):
    loss_grad_avg[i,:] = np.mean(loss_grad[(5*i):(5*i+5),:])
    loss_grad_std[i,:] = np.std(loss_grad[(5*i):(5*i+5),:])
    loss_bc_avg[i,:] = np.mean(loss_bc[(5*i):(5*i+5),:])
    loss_bc_std[i,:] = np.std(loss_bc[(5*i):(5*i+5),:])
    loss_cl_avg[i,:] = np.mean(loss_cl[(5*i):(5*i+5),:])
    loss_cl_std[i,:] = np.std(loss_cl[(5*i):(5*i+5),:])
    loss_fts_cl_avg[i,:] = np.mean(loss_fts_cl[(5*i):(5*i+5),:])
    loss_fts_cl_std[i,:] = np.std(loss_fts_cl[(5*i):(5*i+5),:])
    loss_total_avg[i,:] = np.mean(loss_total[(5*i):(5*i+5),:])
    loss_total_std[i,:] = np.std(loss_total[(5*i):(5*i+5),:])

iteration = np.linspace(1,20001,20001, dtype=int)
string_opt = [r'SGD', r'Adam', r'Momentum', r'NAG']

fig1, ax1 = plt.subplots(1,1, figsize=(3.5,1.5), dpi=300)
fig2, ax2 = plt.subplots(1,1, figsize=(3.5,1.5), dpi=300)
fig3, ax3 = plt.subplots(1,1, figsize=(3.5,1.5), dpi=300)
fig4, ax4 = plt.subplots(1,1, figsize=(3.5,1.5), dpi=300)
fig5, ax5 = plt.subplots(1,1, figsize=(3.5,1.5), dpi=300)

for i in range(4):
    ax1.plot(iteration, loss_grad_avg[i,:], label=string_opt[i], lw=1.0)
    ax2.plot(iteration, loss_bc_avg[i,:], label=string_opt[i], lw=1.0)
    ax3.plot(iteration, loss_cl_avg[i,:], label=string_opt[i], lw=1.0)
    ax4.plot(iteration, loss_fts_cl_avg[i,:], label=string_opt[i], lw=1.0)
    ax5.plot(iteration, loss_total_avg[i,:], label=string_opt[i], lw=1.0)


ax1.set_yscale('log')
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
fig1.xlabel(r'Iteration', fontsize=10)
fig1.ylabel(r'Gradient Loss', fontsize=10)
fig1.tick_params(axis='both', which='major', labelsize=9)
fig1.tick_params(axis='both', which='minor', labelsize=9)
legend=fig1.legend(bbox_to_anchor=(-0.2,1), loc='upper right', fontsize=8, ncol=1, title=r'Optimizer',framealpha=1.0, fancybox=True)
legend.get_title().set_fontsize('8')

ax2.set_yscale('log')
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
fig2.xlabel(r'Iteration', fontsize=10)
fig2.ylabel(r'BC Loss', fontsize=10)
fig2.tick_params(axis='both', which='major', labelsize=9)
fig2.tick_params(axis='both', which='minor', labelsize=9)
legend=fig2.legend(bbox_to_anchor=(-0.2,1), loc='upper right', fontsize=8, ncol=1, title=r'Optimizer',framealpha=1.0, fancybox=True)
legend.get_title().set_fontsize('8')


fig1.show()
fig1.savefig('loss_grad.pdf', bbox_inches='tight')
fig1.close()
fig2.show()
ax2.savefig('loss_bc.pdf', bbox_inches='tight')
ax2.close()
fig3.show()
ax3.savefig('loss_bc.pdf', bbox_inches='tight')
ax3.close()
fig4.show()
ax4.savefig('loss_fts_cl.pdf', bbox_inches='tight')
ax4.close()
fig5.show()
ax5.savefig('loss_total.pdf', bbox_inches='tight')
ax5.close()
