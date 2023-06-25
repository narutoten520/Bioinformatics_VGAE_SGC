# Bioinformatics_VGAE_SGC
基于变分自编码器的空间转录组细胞聚类研究

------
空间转录组测序技术能够在生成基因表达谱的同时，保留细胞在组织内部的位置信息。如何充分利用基因表达谱和空间位置信息来识别空间区域，完成细胞亚群聚类是空间转录组学数据分析的基础和关键。
本文提出基于变分自编码器和图神经网络结合的空间转录组细胞亚群聚类方法。构建双层编码器结构，每一层包含简化图卷积(Simple Graph Convolution, SGC)，用以生成低维表征。
解码器用以重构特征矩阵，通过最小化损失函数来提高低维表征质量。对低维表征进行下游聚类，生成不同的细胞亚群。
提出的聚类方法与多个基准方法在常用的空间转录数据集上进行比较，在聚类准确性和适应性方面都有优势，证明了提出方法的有效性。
## Workflow of spatial clustering task
![](https://github.com/narutoten520/Bioinformatics_VGAE_SGC/blob/e0a33dbc31aa4a754a8cd2ea24bb308c136fd86a/%E5%9B%BE1.png)

## Contents
* [Prerequisites](https://github.com/narutoten520/Benchmark_SRT#prerequisites)
* [Example usage](https://github.com/narutoten520/Benchmark_SRT#example-usage)
* [Benchmarking methods](https://github.com/narutoten520/Benchmark_SRT#benchmarking-methods)
* [Datasets Availability](https://github.com/narutoten520/Benchmark_SRT#datasets-availability)
* [License](https://github.com/narutoten520/Benchmark_SRT#license)
* [Trouble shooting](https://github.com/narutoten520/Benchmark_SRT#trouble-shooting)

### Prerequisites

1. Python (>=3.8)
2. Scanpy
3. Squidpy
4. Pytorch_pyG
5. Pandas
6. Numpy
7. Sklearn
8. Seaborn
9. Matplotlib
10. Torch_geometric

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Example usage
* Selecting the optimal GNN for spatial clustering task in GraphST
  ```sh
    running GraphST.py to choose the suitable GNN for GraphST on human breast cancner data
  ```
* Selecting the optimal GNN for spatial clustering task in STAGATE
  ```sh
    running STAGATE.py to choose the suitable GNN for GraphST on human breast cancner data
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Benchmarking methods
Benchmarking methods used in this paper include: 
* [CCST](https://github.com/xiaoyeye/CCST)
* [DeepST](https://github.com/JiangBioLab/DeepST)
* [GraphST](https://github.com/JinmiaoChenLab/GraphST)
* [STAGATE](https://github.com/zhanglabtools/STAGATE)
* [SpaGCN](https://github.com/jianhuupenn/SpaGCN)
* [conST](https://github.com/ys-zong/conST)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Datasets Availability

* [Human DLPFC](https://github.com/LieberInstitute/spatialLIBD)
* [Mouse brain](https://squidpy.readthedocs.io/en/stable/auto_tutorials/tutorial_visium_hne.html)
* [Slide-seqV2](https://squidpy.readthedocs.io/en/stable/auto_tutorials/tutorial_slideseqv2.html)
* [Stereo-seq](https://stagate.readthedocs.io/en/latest/T4_Stereo.html)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Trouble shooting

* data files<br>
Please down load the spatial transcriptomics data from the provided links.

* Porch_pyg<br>
Please follow the instruction to install pyG and geometric packages.
