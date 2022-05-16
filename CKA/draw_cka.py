import numpy as np
from CKA import *
import cv2
import matplotlib.pyplot as plt

feature_maps_one_epoch=np.load("fms_oneepoch.npy", allow_pickle=True)

for feature_maps_one_batch in feature_maps_one_epoch:
    scores=np.zeros((len(feature_maps_one_batch), len(feature_maps_one_batch)))
    for i in range(len(feature_maps_one_batch)):
        for j in range(len(feature_maps_one_batch)):
            fmx=feature_maps_one_batch[i].cpu().numpy()
            fmy=feature_maps_one_batch[j].cpu().numpy()

            scores[i][j]=kernel_CKA(fmx, fmy)
            
    print(scores)
    ax = plt.matshow(scores)
    plt.colorbar(ax.colorbar, fraction=0.025)
    plt.title("matrix X")
    plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# X=np.array([[0,3,2,4],[5,4,7,8],[9,16,8,5],[13,3,4,16],[6,18,1,20]])
# #A = np.arange(0, 100).reshape(10, 10)
# ax = plt.matshow(X)
# plt.colorbar(ax.colorbar, fraction=0.025)
# plt.title("matrix X")
# plt.show()