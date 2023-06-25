import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data

def Adata2Torch_data(adata): #提取adata中的features 和 edge_index, 即基因表达矩阵和邻接矩阵。
    G_df = adata.uns['Spatial_Net'].copy() #spoti and spotj之间的距离
    spots = np.array(adata.obs_names) #Spot_names
    spots_id_tran = dict(zip(spots, range(spots.shape[0]))) #对每个Spot进行编号
    G_df['Spot1'] = G_df['Spot1'].map(spots_id_tran) 
    G_df['Spot2'] = G_df['Spot2'].map(spots_id_tran) # 构建num_edges的过程。

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Spot1'], G_df['Spot2'])), shape=(adata.n_obs, adata.n_obs))
    # n_obs 是Spot个数；G是n_obs * n_obs的稀疏矩阵
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G) # edgeList就是构建出来的torch_geometric 中的num_edges
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data
    # data的数据形式如下：Data(x=[3460, 33538], edge_index=[2, 23512])
    # 根据挑选出来的邻居构建邻接矩阵和基因表达数据。

def Spatial_Dis_Cal(adata, rad_dis=None, knn_dis=None, model='Radius', verbose=True):
    """\
    Calculate the spatial neighbor networks, as the distance between two spots.
    Parameters
    ----------
    adata:  AnnData object of scanpy package.
    rad_dis:  radius distance when model='Radius' 半径
    knn_dis:  The number of nearest neighbors when model='KNN' 邻居个数
    model:
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_dis. 
        When model=='KNN', the spot is connected to its first knn_dis nearest neighbors.
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    assert(model in ['Radius', 'KNN']) #断言语句，可以用来调试程序。
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial']) #Spot 空间坐标
    coor.index = adata.obs.index #df的index改为spot名称
    # coor.columns = ['imagerow', 'imagecol']
    coor.columns = ['Spatial_X', 'Spatial_Y'] #修改df的列名

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_dis).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        # Find the neighbors within a given radius of a point or points, 返回每个Spot在给定半径中的邻居个数及距离。
        # distances, indices的rows等于Spot的个数，即每个Spot都对应一个distance 和 index list.
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices[spot].shape[0], indices[spot], distances[spot]))) #每个spot的邻居编号；距离。
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=knn_dis+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices.shape[1],indices[spot,:], distances[spot,:])))

    KNN_df = pd.concat(KNN_list) #变为dataframe格式。
    KNN_df.columns = ['Spot1', 'Spot2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_spot_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), )) #构建一个词典，
    Spatial_Net['Spot1'] = Spatial_Net['Spot1'].map(id_spot_trans) #Spot1的编号，e.g. spot1出现几次，表明有几个邻居，在spot2里。
    Spatial_Net['Spot2'] = Spatial_Net['Spot2'].map(id_spot_trans) #Spot2的编号 spot1对应的邻居编号
    if verbose:
        print('The graph contains %d edges, %d spots.' %(Spatial_Net.shape[0], adata.n_obs)) #共多少条边
        print('%.4f neighbors per spot on average.' %(Spatial_Net.shape[0]/adata.n_obs)) #平均每个Spot多少条边 

    adata.uns['Spatial_Net'] = Spatial_Net

def Spatial_Dis_Draw(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Spot1'].shape[0] #共多少条边
    Mean_edge = Num_edge/adata.shape[0] #平均多少条边
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Spot1'])) 
    plot_df = plot_df/adata.shape[0]  #plot_df是一个概率list, 和为1.
    fig, ax = plt.subplots(figsize=[4,4],dpi=300)
    plt.ylabel('Percentage')
    plt.xlabel('Edge Numbers per Spot')
    plt.title('Number of Neighbors for Spots (Average=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df,color="#aa40fc",edgecolor="#f7b6d2",linewidth=2)

def Cal_Spatial_variable_genes(adata):
    import SpatialDE
    # counts = pd.DataFrame(adata.X.todense(), columns=adata.var_names, index=adata.obs_names)
    counts = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
    coor = pd.DataFrame(adata.obsm['spatial'], columns=['Spatial_X', 'Spatial_Y'], index=adata.obs_names)
    Spatial_var_genes = SpatialDE.run(coor, counts)
    Spatial_3000_var_genes = Spatial_var_genes["g"].values[0:3000]
    Spatial_3000_var_genes = pd.DataFrame(Spatial_3000_var_genes)
    all_genes = counts.columns.to_frame()
    for i in range(len(all_genes.values)):
        if all_genes.values[i] in Spatial_3000_var_genes.values:
            all_genes.values[i] =1
        else:
            all_genes.values[i] =0
    Spatial_highly_genes = all_genes.squeeze()
    adata.var["Spatial_highly_variable_genes"] = Spatial_highly_genes.astype(bool)

def DGI_loss_Draw(adata):
    import matplotlib.pyplot as plt
    if "Model_loss" not in adata.uns.keys():
        raise ValueError("Please Train Graph Deep Learning Model first!") #验证是否存在Spatial_Net
    Train_loss = adata.uns["Model_loss"]
    plt.style.use('default') #'seaborn-poster';seaborn-paper';'seaborn-deep''ggplot'
    plt.plot(Train_loss,label='Training loss',linewidth=2)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss of pyG model")
    plt.legend()
    plt.grid()

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata
