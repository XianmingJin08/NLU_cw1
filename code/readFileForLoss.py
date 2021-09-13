import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plotHeatMap (datadir, keys,q):
    dict = np.load((datadir+".npy"),allow_pickle='TRUE').item()
    uniqueB = []
    for i,(a,b,c) in enumerate(dict.keys()):
        if b not in uniqueB:
            uniqueB.append(b)

    for b in uniqueB:
        df = pd.DataFrame(index=range(0, 30), columns=keys)
        for i, (aa, bb, cc) in enumerate(dict.keys()):
            df.iloc[i] = ([aa,bb,cc,float(dict[aa,bb,cc])])

        p2 = keys[1]
        df = df[df[p2]==b]
        df = df.pivot(keys[0], keys[2], keys[3])
        index_tmp = (df.index)
        column_tmp = (df.columns)
        df = df.to_numpy()
        a = []
        for i in df:
            tmp = []
            for j in i:
                tmp.append(j)
            a.append(tmp)
        df = pd.DataFrame(a, index=index_tmp, columns=column_tmp)
        ax = sns.heatmap(df, cmap='binary', annot=True, fmt='.4g', cbar_kws={'label': keys[3]})
        ax.set_title(keys[1]+'='+str(b))
        figure = ax.get_figure()
        figure.savefig(q+keys[1]+str(b))
        figure.clf()


if __name__ == "__main__":

    # keys = [(0.5, 25, 0), (0.5, 25, 2), (0.5, 25, 5), (0.5, 50, 0), (0.5, 50, 2), (0.5, 50, 5), (0.1, 25, 0), (0.1, 25, 2), (0.1, 25, 5), (0.1, 50, 0), (0.1, 50, 2), (0.1, 50, 5), (0.05, 25, 0), (0.05, 25, 2), (0.05, 25, 5), (0.05, 50, 0), (0.05, 50, 2), (0.05, 50, 5)]
    # index = 0
    # dict3 = {}
    # with open('Result3.txt', 'r') as file:
    #     for line in file:
    #         if 'Accuracy:' in line:
    #             print (line.split()[1])
    #             dict3[keys[index]] = line.split()[1]
    #             index += 1
    #             if (index == len(keys)):
    #                 break
    # print (dict3)
    # np.save ("dict3",dict3)
    # dict2a = np.load('dict2a.npy', allow_pickle='TRUE').item()
    # print (dict2a)
    # df = pd.DataFrame(index=range(0,18),columns=['lr','hdim','steps','loss'])
    # for i,(a,b,c) in enumerate(dict2a.keys()):
    #     df.iloc[i] = ([a,b,c,float(dict2a[a,b,c])])
    #
    # df = df[df['hdim']==50]
    # df = df.drop(['hdim'],axis=1)
    # df = df.pivot("lr", "steps", "loss")
    # index_tmp = (df.index)
    # column_tmp = (df.columns)
    # print (column_tmp)
    # print (df)
    # df = df.to_numpy()
    # print (df)
    # a = []
    # for i in df:
    #     tmp = []
    #     for j in i:
    #         print (j)
    #         tmp.append(j)
    #     a.append(tmp)
    # df = pd.DataFrame (a,index=index_tmp,columns=column_tmp)
    # ax = sns.heatmap(df,annot=True,fmt='.4g',cbar_kws={'label': 'mean loss'})
    # ax.set_title('H=50')
    # figure = ax.get_figure()
    # figure.savefig('2aH50.png')

    # plotHeatMap("dict2aExtra",['anneal','batch_size','min_change','loss'],'2aExtra')
    # plotHeatMap("dict2a",['lr','hdim','steps','loss'],'2a')
    # plotHeatMap("dict3",['lr','hdim','steps','acc'],'3')
    # plotHeatMap("dict_acc_b",['lr','hdim','steps','acc'],'5b')
    # plotHeatMap("dict_acc_c",['anneal','batch_size','min_change','acc'],'5c')

    # dict5a = np.load('dict_acc_a.npy', allow_pickle='TRUE').item()
    # xs = list(dict5a.keys())
    # ys = list(dict5a.values())
    # plt.plot(xs,ys)
    # plt.xlabel("vocab size")
    # plt.ylabel("accuracy")
    # plt.title("The plot shows how accuracy is affected by vocab size")
    # plt.savefig("5a")

    # dict5b_acc = np.load('dict_acc_b.npy', allow_pickle='TRUE').item()
    # dict5b_loss = np.load('dict_loss_b.npy', allow_pickle='TRUE').item()
    # xs_5b_acc = range(0,len(dict5b_acc.keys()))
    # ys_5b_acc = np.array(list(dict5b_acc.values()))
    # xs_5b_loss = range(0, len(dict5b_loss.keys()))
    # ys_5b_loss = np.array(list(dict5b_loss.values()))
    # ys_5b_loss = ys_5b_loss[ys_5b_acc.argsort()]
    # ys_5b_acc = ys_5b_acc [ys_5b_acc.argsort()]
    # fig, ax_left = plt.subplots()
    # ax_right = ax_left.twinx()
    # ax_left.plot(xs_5b_acc,ys_5b_acc, color='blue',label='accuracy')
    # ax_right.plot(xs_5b_loss,ys_5b_loss, color='red',label='loss')
    # ax_left.set_xlabel("different settings")
    # ax_left.set_ylabel("accuracy")
    # ax_right.set_ylabel("loss")
    # ax_left.legend()
    # ax_right.legend()
    # plt.title("The plot shows how accuracy and mean loss changes in different setting")
    # plt.savefig("5blossAcc")

    dict5b_acc = np.load('dict_acc_a.npy', allow_pickle='TRUE').item()
    dict5b_loss = np.load('dict_loss_a.npy', allow_pickle='TRUE').item()
    xs_5b_acc = [1000,2000,3000,4000,5000,6000,7000,8000,9000]
    ys_5b_acc = np.array(list(dict5b_acc.values()))
    ys_5b_loss = np.array(list(dict5b_loss.values()))
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    ax_left.plot(xs_5b_acc,ys_5b_acc, color='blue',label='accuracy')
    ax_right.plot(xs_5b_acc,ys_5b_loss, color='red',label='loss')
    ax_left.set_xlabel("vocab size")
    ax_left.set_ylabel("accuracy")
    ax_right.set_ylabel("loss")
    ax_left.legend()
    ax_right.legend()
    plt.title("The plot of accuracy and mean loss changes with different vocab size")
    plt.savefig("5alossAcc")

