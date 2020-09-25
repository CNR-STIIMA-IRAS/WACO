import re
import os
import numpy
from pprint import pprint
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D


def spec_aver(list_el):
    len_lis = len(list_el)
    avr = 0.0
    for elem in list_el:
        if elem != -1.0:
            avr = avr + elem
        else:
            len_lis = len_lis - 1
    if len_lis != 0.0:
        return avr / len_lis
    else:
        return 0.0000001


def moving_average(data, window=3):
    avg = numpy.zeros_like(data, dtype='float')
    for i in range(window):
        avg[i] = numpy.mean(list(filter((numpy.nan).__ne__, data[0:i + 1])))

    for i in range(window, len(data)):
        avg[i] = numpy.mean(list(filter((numpy.nan).__ne__, data[i - window + 1: i + 1])))

    return avg


def BestSoFar(data):
    bsf = list()
    for i in range(len(data[0, :])):
        value = max(data[:, i])
        if i == 0:
            if value > -1.0:
                bsf.append(value)
            else:
                bsf.append(-1.0)
        else:
            if value > bsf[i - 1]:
                bsf.append(value)
            else:
                bsf.append(bsf[i - 1])

    return bsf


expStruct = namedtuple("ExpStruct", "Whales Whales_iter \
                                     Ants Ants_iter \
                                     N_open_cone N_rot_cone \
                                     exec_time fobj max_obj type")

PATH = os.getcwd()
PATH = os.path.join(PATH, "output")
# FOLDS = ["asola_tubo_storto_51"]
FOLDS = ["asola_tubo_storto_51"]
expList = []
whale_numL = []
whale_iteL = []
ants_iterL = []
open_coneL = []
rot_coneL = []
# estrai i parametri dai nomi delle cartelle in una lista di namedtouple e raccogli i risulatati dai file
for i, it in enumerate(FOLDS):
    currfold = os.path.join(PATH, FOLDS[i])
    print("PATH:", currfold)
    for path in os.walk(currfold):
        for fol in path[1]:
            curr = os.path.join(path[0], fol)
            for pd in os.walk(curr):
                if len(pd[1]) > 0:
                    for same in pd[1]:
                        fin = os.path.join(curr, same)
                        extime = open(os.path.join(fin, "execution_time.txt"))
                        fobj = numpy.loadtxt(open(os.path.join(fin, "fobj_history.txt")), delimiter=",", unpack=True)
                        best = open(os.path.join(fin, "score.txt"))
                        params = fol.split("_", 7)
                        if params[0] not in whale_numL:
                            whale_numL.append(params[0])
                        if params[1] not in whale_iteL:
                            whale_iteL.append(params[1])
                        if params[5] not in open_coneL:
                            open_coneL.append(params[5])
                        if params[6] not in rot_coneL:
                            rot_coneL.append(params[6])
                        expList.append(expStruct(params[0], params[1], params[2], params[3], \
                                                 params[5], params[6], \
                                                 float(extime.read()), fobj, float(best.read()), FOLDS[i]))

whale_numL = [int(i) for i in whale_numL]
whale_iteL = [int(i) for i in whale_iteL]
open_coneL = [int(i) for i in open_coneL]
rot_coneL = [int(i) for i in rot_coneL]

whale_numL.sort()
whale_iteL.sort()
open_coneL.sort()
rot_coneL.sort()

# print (expList)
# import sys
# sys.exit(0)
# for i, whnum in enumerate(whale_numL):
#     for j, whiter in enumerate(whale_iteL):
#         for c, op_c in enumerate(open_coneL):
#             for r, rot_c in enumerate(rot_coneL):
#                 cnt = 0
#                 for test in expList:
#                     if test.Whales == whnum and test.Whales_iter == whiter and test.N_open_cone == op_c and test.N_rot_cone == rot_c:
#                         cnt = cnt + 1
#                 if cnt != 0:
#                     fig, ax = plt.subplots(nrows=cnt, ncols=1)
#                 else:
#                     continue
#                 cnt2 = 0
#                 for test in expList:
#                     list = []
#                     if test.Whales == whnum and test.Whales_iter == whiter and test.N_open_cone == op_c and test.N_rot_cone == rot_c:
#                         for ind in range(0, int(test.Whales_iter), 1):
#                             list.append(spec_aver(test.fobj[:, ind]))
#                         fig.suptitle("Test : " + str(FOLDS) + ", Whales : " + str(whnum) + ", Open_cone : " + str(op_c) + ", Rot_cone : " + str(rot_c), fontsize=10)
#                         axis = numpy.arange(0, len(test.fobj[0]), 1)
#                         regr = linear_model.LinearRegression()
#                         regr.fit(axis.reshape(-1, 1), list)
#                         line_r = regr.predict(axis.reshape(-1, 1))
#                         print regr.coef_
#                         ax[cnt2].scatter(axis, list)
#                         ax[cnt2].plot(axis, line_r, color='red', linewidth=1.5)
#                         ax[cnt2].set_ylim(0.0, numpy.amax(test.fobj))
#                         cnt2 = cnt2 + 1

# fig = plt.figure()
# ax3d = fig.add_subplot(111, projection='3d')

# dividi e stampa i parametri di esprimenti con stessi parametri e diverso numero di iterazioni delle balene

# for i, whnum in enumerate(whale_numL):
#     for r, rot_c in enumerate(rot_coneL):
#         for c, op_c in enumerate(open_coneL):
#             for test in expList:
#                 if test.Whales == whnum and test.N_open_cone == op_c and test.N_rot_cone == rot_c:
#                     fig, ax = plt.subplots(nrows=2, ncols=1)
#                     break
#             for test in expList:
#                 list = []
#                 if test.Whales == whnum and test.N_open_cone == op_c and test.N_rot_cone == rot_c:
#                     for ind in range(0, int(test.Whales_iter), 1):
#                         # media dei risultati della funz obiettivo per ogni balena ad ogni iterazione
#                         list.append(spec_aver(test.fobj[:, ind]))
#                     fig.suptitle("Test : " + str(FOLDS) + ", Whales : " + str(whnum) + ", Open_cone : " + str(
#                         op_c) + ", Rot_cone : " + str(rot_c), fontsize=10)
#                     axis = numpy.arange(0, len(test.fobj[0]), 1)
#
#                     # regessione del log media della fun obiett media sul numero iterazioni
#
#                     regr = linear_model.LinearRegression()
#                     regr.fit(axis.reshape(-1, 1), numpy.log(list))
#                     ax[0].scatter(float(test.Whales_iter), float(regr.coef_) * float(test.Whales_iter), color='k')
#                     # ax[0].scatter(float(test.Whales_iter), float(popt[0]) * float(test.Whales_iter),color='k')
#                     ax[1].scatter(float(test.Whales_iter), float(test.max_obj), color='red')
#                     ax3d.scatter(float(whnum), float(test.Whales_iter), numpy.log(float(test.max_obj)), marker='^')
#
# matplotlib.pyplot.show()

max_fobj = numpy.zeros(shape=(5, len(whale_numL), len(whale_iteL)), dtype='float')
mean_max_fobj = numpy.zeros(shape=(len(whale_numL), len(whale_iteL)), dtype='float')
std_max_fobj = numpy.zeros_like(mean_max_fobj)

for w, whales in enumerate(whale_numL):
    for it, w_iter in enumerate(whale_iteL):

        fobj = numpy.zeros(shape=(5, whales, w_iter), dtype='float')
        sample = 0

        plt.figure()

        for test in expList:
            if int(test.Whales) == whales and int(test.Whales_iter) == w_iter \
                    and int(test.N_rot_cone) == rot_coneL[-1] and int(test.N_open_cone) == open_coneL[-1]:
                fobj[sample, :, :] = test.fobj
                max_fobj[sample, w, it] = test.max_obj
                sample = sample + 1

        fobj_mean = numpy.zeros(shape=(sample, w_iter))
        fobj_std = numpy.zeros_like(fobj_mean, dtype='float')
        fobj_mov_avg = numpy.zeros_like(fobj_mean, dtype='float')
        best_so_far = numpy.zeros_like(fobj_mean)

        mean_max_fobj[w, it] = numpy.mean(max_fobj[:, w, it])
        std_max_fobj[w, it] = numpy.std(max_fobj[:, w, it])

        for i in range(sample):
            for iter in range(w_iter):
                fobj_mean[i, iter] = numpy.mean(list(filter((-1.0).__ne__, fobj[i, :, iter])))
                fobj_std[i, iter] = numpy.std(list(filter((-1.0).__ne__, fobj[i, :, iter])))

            best_so_far[i, :] = BestSoFar(fobj[i, :, :])
            fobj_mov_avg[i, :] = moving_average(fobj_mean[i, :], window = 5)

            plt.plot(numpy.arange(1, w_iter + 1), fobj_mov_avg[i,:])
            plt.plot(numpy.arange(1, w_iter + 1), best_so_far[i, :], ':')
        #
        # # plt.plot(numpy.arange(1, w_iter + 1), numpy.mean(fobj_mov_avg, axis=0))
        #
        plt.plot(numpy.arange(1, w_iter + 1), numpy.mean(best_so_far, axis = 0), '--')
        plt.title('Whales: ' + str(whales) + ' Iterations: ' + str(w_iter))

#
# plt.figure()
# plt.imshow(mean_max_fobj)
# plt.axis
#
# plt.figure()
# plt.figure(std_max_fobj)

fig, ax = plt.subplots(nrows=1, ncols=2)

im1 = ax[0].imshow(mean_max_fobj, cmap='jet')
fig.colorbar(im1, ax=ax[0])
ax[0].set_title('Mean max f_obj')

im2 = ax[1].imshow(std_max_fobj, cmap='jet')
fig.colorbar(im2, ax=ax[1])
ax[1].set_title('Std max f_obj')
