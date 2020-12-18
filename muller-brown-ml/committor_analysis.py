import numpy as np

for i in range(4,11):
    print(i)
    validation_score = np.zeros((1,))
    data_nn = np.zeros((1,))
    data_em = np.zeros((1,))
    for j in range(1,13):
        data = np.genfromtxt(str(i)+"/simple_validation_"+str(j)+".txt", usecols=(0,1))
        data_nn = np.append(data_nn, data[:,0])
        data_em = np.append(data_em, data[:,1])
        data = np.abs(data[:,0]-data[:,1])
        validation_score = np.append(validation_score,1-data)
    validation_score = validation_score[1:]
    val_mean = validation_score.mean()
    val_std = 1.96*np.std(validation_score)/len(validation_score)**0.5
    print(str(data_nn.mean())+" "+str(data_em.mean()))
    print("{:.8g} {:.8g}".format(val_mean,val_std))
