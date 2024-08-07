---
title: '从曲线图中提取坐标数据'
date: 2022-01-02
permalink: /posts/2022/01/blog-post-1/
tags:
  - Python
  - 论文
---

今天说一下从论文中的曲线数据中提取原始坐标点数据的方法。

如下图所示

![](https://borninfreedom.github.io/images/blog2022/origin.png)

我们想要提取这个图中的两条曲线的原始数据，也就是他们的坐标点。

我们需要借助一个软件，[Engauge Digitizer](https://markummitchell.github.io/engauge-digitizer/)，到官网后选择Latest Release下载即可。

![](https://borninfreedom.github.io/images/blog2022/engauge.png)


安装完成之后，我们导入要提取数据的图片。

![](https://borninfreedom.github.io/images/blog2022/import.png)

中间呢有一个对话框让我们输入要处理的曲线名字，那我这里就按照原始图像中的名字来命名两条曲线。如图

![](https://borninfreedom.github.io/images/blog2022/name.png)

导入进来之后呢，注意选择Original image，如图

![](https://borninfreedom.github.io/images/blog2022/originalimage.png)

下面要做的就是添加坐标系的坐标，主要作用就是给提取数据建立一个虚拟的坐标系，好与原始的数据在坐标系上对其。

![](https://borninfreedom.github.io/images/blog2022/axis.png)

首先找到原图中的坐标系原点，在这个地方添加一个坐标系的点

![](https://borninfreedom.github.io/images/blog2022/addaxis1.png)

依次添加

![](https://borninfreedom.github.io/images/blog2022/addaxis2.png)
![](https://borninfreedom.github.io/images/blog2022/addaxis3.png)

添加点的时候尽量把原始图片放大一些，精准放置新的坐标点。放置3个点就可以了，软件会自动添加上第四个点。

添加好的坐标系的点如图所示

![](https://borninfreedom.github.io/images/blog2022/allaxis.png)

下面就是要提取曲线的数据了，首先切换到提取点工具

![](https://borninfreedom.github.io/images/blog2022/point.png)

然后要选择要提取的曲线

![](https://borninfreedom.github.io/images/blog2022/point1.png)

然后在要提取的曲线上添加点就可以了，较弯曲的地方可以多放一些点，平滑的地方少放一些点，如下图所示

![](https://borninfreedom.github.io/images/blog2022/allpoints.png)

然后切换到另外一条曲线

![](https://borninfreedom.github.io/images/blog2022/alpha01.png)

重复提取点的步骤，提取完两条曲线，如下图

![](https://borninfreedom.github.io/images/blog2022/allpoints01.png)

这时候，就可以把数据导出了

![](https://borninfreedom.github.io/images/blog2022/export.png)

导出为一个csv文件

![](https://borninfreedom.github.io/images/blog2022/csv.png)

用excel打开文件看一下

![](https://borninfreedom.github.io/images/blog2022/csvfile.png)

下面直接在excel里把曲线画出来也可以，编程画也可以，这里演示一下编程画图，我将导出的文件命名为了printscreen.csv

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('printscreen.csv')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

x = data['x'].to_numpy()
y1 = data['alpha20'].to_numpy()
y2 = data['alpha0.1'].to_numpy()

ax.plot(x,y1,'g-')
ax.plot(x,y2,'b--')
```

![](https://borninfreedom.github.io/images/blog2022/jupyter.png)

看一下是不和原图一样呢