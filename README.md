# U-GAT-IT 百度飞桨论文复现
百度飞桨论文复现Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation
1. [论文心得](https://github.com/liuxianyi/U-GAT-IT/blob/master/README.md#%E8%AE%BA%E6%96%87%E5%BF%83%E5%BE%97)
2. 论文PaddlePaddle([百度飞桨](https://aistudio.baidu.com/aistudio/))[复现](https://github.com/liuxianyi/U-GAT-IT/blob/master/README.md#复现)

## 论文心得
### 摘要
  论文提出了一个新的方法在无监督条件下对图像实现意象到意象的转换，将注意力模块与新的可自主学习的归一化(normalization)相结合实现不错的效果。与传统的state-of-the-art模型项目，改论文提出的方法具有很大的优势。
  注意力模块:更加关注区域的重要性，来区别源图与目标图。它能够把握住域之间的几何改变（整体改变，形状的改变）。
  AdaLIN自适应层级归一化函数：帮助我们在不改变网络架构的情况下，灵活控制形状改变的数目，以及纹理的改变。
### 实现的方案
1. 整体结构由两个生成器(G(s->t)生成器1：原图像->目标图像；G(t->s)生成器2：目标图像->原图像。他们类似于cycle gan的操作）和两个判别器组成，并将注意力模块集成到两个生成器(attention model作用：不同域生成不同效果）和两个判别器$D_t、D_s$（attention model作用：指引生成器在指定区域生成逼真图片）。
2. 训练生成器 $G_{s->t})$
3.  网络的结构

### 评估的效果


### 应用
### 总结
## 复现
