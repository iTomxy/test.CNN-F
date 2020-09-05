import os
import numpy as np
import scipy.io as sio


"""compare `vgg_net.mat` and `imagenet-vgg-f.mat`
conclusion: they are the SAME
"""


P = "G:/dataset"
VGG_NET = os.path.join(P, "vgg_net.mat")
VGG_F = os.path.join(P, "imagenet-vgg-f.mat")

vgg_net = sio.loadmat(VGG_NET)
vgg_net = vgg_net["net"][0][0][0][0]

vgg_f = sio.loadmat(VGG_F)
vgg_f = vgg_f["layers"][0]

for i in range(len(vgg_net)):
    print("--- layer:", i + 1, "---")
    lay_n = vgg_net[i][0][0]  # vgg_net
    lay_f = vgg_f[i][0][0]  # imagenet-vgg-f
    print("name:", lay_n[0], lay_f[0])
    print("type:", lay_n[1], lay_f[1])
    _type = lay_n[1]
    
    if "conv" == _type:
        kn, bn = lay_n[2][0]
        kf, bf = lay_f[2][0]
        diff_k = (kn != kf).astype(np.int).sum()
        diff_b = (bn != bf).astype(np.int).sum()
        print("kernel:", kn.shape, kf.shape, diff_k)
        print("bias:", bn.shape, bf.shape, diff_b)
        print("pad:", lay_n[4], lay_f[4])
        print("stride:", lay_n[5], lay_f[5])
    elif "relu" == _type:
        pass
    elif "lrn" == _type:
        pn = lay_n[2][0]
        pf = lay_f[2][0]
        print("local_size/depth_radius:", pn[0], pf[0])
        print("bias:", pn[1], pf[1])
        print("alpha:", pn[2], pf[2])
        print("beta:", pn[3], pf[3])
    elif "pool" == _type:
        print("pool type:", lay_n[2], lay_f[2])
        print("size:", lay_n[3], lay_f[3])
        print("stride:", lay_n[4], lay_f[4])
        print("pad:", lay_n[5], lay_f[5])
