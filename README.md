SIFT算法简明文档  
====  
尺度不变特征转换(Scale-invariant feature transform或SIFT)是一种用于侦测与描述影像中的局部性特征的计算机视觉算法。该算法在空间尺度中寻找极值点，并提取出其位置、尺度、旋转不变量，最早由 David Lowe于1999年所发表，2004年完善总结。本文根据论文《Distictive Image Features From Scale Invariant Keypoints》（Lowe，2004）及其在OpenCV中的实现源码，对算法基本数学原理做简要说明。    
 
1. 基本流程  
----    
Lowe将SIFT算法分解为如下四步：    
1. 尺度空间极值检测：搜索所有尺度上的图像位置。通过高斯微分函数来识别潜在的对于尺度和旋转不变的兴趣点。  
2. 关键点定位：在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度。   
3. 方向确定：基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而提供对于这些变换的不变性。   
4. 关键点描述：在每个关键点周围的邻域内，在选定的尺度上测量图像局部的梯度。这些梯度被变换成一种表示，这种表示允许比较大的局部形状的变形和光照变化。    


2. 高斯模糊  
----
由上可知SIFT算法是在不同的尺度空间上查找关键点，而尺度空间的获取需要使用高斯模糊来实现，Lindeberg等人已证明高斯卷积核是实现尺度变换的唯一变换核，并且是唯一的线性核。       

**2.1 高斯模糊函数**    
高斯模糊是一种图像滤波器，它使用正态分布(高斯函数)计算模糊模板，并使用该模板与原图像做卷积运算，达到模糊图像（滤波）目的。     
N维空间的正态分布方程：        
<img src="http://www.forkosh.com/mathtex.cgi? \Medium G(r) = \frac{1}{{{{\sqrt {2\pi {\sigma ^2}} }^N}}}{e^{ - {r^2}/(2{\sigma ^2})}}">    *(eq.1)*       
其中：     
<img src="http://www.forkosh.com/mathtex.cgi? \Medium \sigma"> 为正态分布标准差，该值越大则图像越模糊（平滑）；  
<img src="http://www.forkosh.com/mathtex.cgi? \Medium r"> 为模糊半径，即模板元素到模板中心的距离。  
对于二维图像，假设其模板大小为 <img src="http://www.forkosh.com/mathtex.cgi? \Medium m*n"> ，则模板元素 <img src="http://www.forkosh.com/mathtex.cgi? \Medium (x,y)"> 的计算公式为：    

<img src="http://www.forkosh.com/mathtex.cgi? \Medium G(x,y) = \frac{1}{{2\pi {\sigma ^2}}}{e^{ - \frac{{{{(x - m/2)}^2} + {{(y - n/2)}^2}}}{{2{\sigma ^2}}}}}">  *(eq.2)*     

<img src="http://my.csdn.net/uploads/201204/29/1335629457_4620.jpg">    

在二维空间中，该公式对应的曲面等高线是从中心开始呈正态分布的同心圆，分布不为零的像素组成的模板矩阵与原始图像进行卷积变换。理论上模板中各像素的分布都不为零，模板大小应与变换图像相同，但实际应用中，计算高斯函数的离散近似时可忽略距离 <img src="http://www.forkosh.com/mathtex.cgi? \Medium 3\sigma"> 之外的像素影响。通常只需计算 <img src="http://www.forkosh.com/mathtex.cgi? \Medium  (6\sigma+1)\times(6\sigma+1) "> 的模板即可保证相关像素计算的完整性。例如设 <img src="http://www.forkosh.com/mathtex.cgi? \Medium \sigma=0.6"> ，则模板大小为5 * 5，按下式计算归一化常数：     

<img src="http://www.forkosh.com/mathtex.cgi? \Medium U(x,y) = \sum\limits_1^m {\sum\limits_1^n {G(x,y)} } ">    

则归一化后的高斯模板矩阵各元素如下：  

| 5*5 Matrix | Colume1  | Colume2  | Colume3  | Colume4  | Colume5  |
| ---- |:-------- |:-----------:|:-----------:|:-----------:| --------:|
| **Row1** | 6.59E-06 | 0.000424781 | 0.001703536 | 0.000424781 | 6.59E-06 | 
| **Row2** | 0.000424781 | 0.027398382 | 0.109878241 | 0.027398382 | 0.000424781 | 
| **Row3** | 0.001703536 | 0.109878241 | 0.440654774 | 0.109878241 | 0.001703536 | 
| **Row4** | 0.000424781 | 0.027398382 | 0.109878241 | 0.027398382 | 0.000424781 | 
| **Row5** | 6.59E-06 | 0.000424781 | 0.001703536 | 0.000424781 | 6.59E-06 | 



**2.2 图像卷积操作**    
设输入图像为<img src="http://www.forkosh.com/mathtex.cgi? \Medium I">，输出图像为<img src="http://www.forkosh.com/mathtex.cgi? \Medium O">，根据 <img src="http://www.forkosh.com/mathtex.cgi? \Medium \sigma"> 使用 *(eq.2)* 计算各模板元素的值并做归一化处理得到高斯模板矩阵后，按以下公式对原始图像做卷积：    
<img src="http://www.forkosh.com/mathtex.cgi? \Medium O(i,j)=G(x,y,\sigma)\otimes I(x,y) ">        *(eq.3)*   
即：     
<img src="http://www.forkosh.com/mathtex.cgi? \Medium O(i,j) = \sum\limits_{x = 1}^m {\sum\limits_y^n {G(x,y)} } I(i + x,j + y)">      *(eq.4)*   

其示意图如下： 
<img src="http://my.csdn.net/uploads/201204/29/1335629489_1963.jpg" />    

实现过程中，为减少卷积计算强度，可采用高斯模板的分离形式简化。高斯函数的可分离性是指使用二维矩阵变换的结果等效于在水平方向进行一维高斯矩阵变换的结果再进行竖直方向的一维高斯矩阵变换得到。从计算的角度来看，这是一项有用的特性，可将原有的复杂度 <img src="http://www.forkosh.com/mathtex.cgi? \Medium O(m \times n \times M \times N)"> 减少为 <img src="http://www.forkosh.com/mathtex.cgi? \Medium O(m \times M \times N) + O(n \times M \times N)">，详见[二维高斯模糊和可分离核形式的快速实现](http://m.blog.csdn.net/blog/zxpddfg/45912561)      
高斯模糊的实现方式可参考**OpenCV的`cv::GaussianBlur`方法**。        

3. 尺度空间极值检测
----
尺度空间使用高斯金字塔表示。Tony Lindeberg指出尺度规范化的LoG(Laplacion of Gaussian)算子具有真正的尺度不变性，Lowe使用高斯差分金字塔近似LoG算子，在尺度空间检测稳定的关键点。    

**3.1 尺度空间的表示**    
一个图像的尺度空间 <img src="http://www.forkosh.com/mathtex.cgi? \Medium  L(x,y,\sigma) "> 定义为一个变化尺度的高斯函数 <img src="http://www.forkosh.com/mathtex.cgi? \Medium  G(x,y,\sigma) "> 与原图像 <img src="http://www.forkosh.com/mathtex.cgi? \Medium I(x,y)"> 的卷积:    
<img src="http://www.forkosh.com/mathtex.cgi? \Medium L(x,y,\sigma)= G(x,y,\sigma)\otimes(I(x,y) ">     
与公式*(eq.3)*相同。这里<img src="http://www.forkosh.com/mathtex.cgi? \Medium \sigma"> 称为尺度空间因子，值越小表示图像被平滑的越少，相应的尺度也就越小。大尺度对应于图像的概貌特征，小尺度对应于图像的细节特征。    

**3.2 高斯金字塔的构建**    
尺度空间在实现时使用高斯金字塔表示，高斯金字塔的构建分为两部分：
1. 生成多个组（Octave）的降采样图像，即某一组的图像由其前一组做降采样(隔点采样)得到；    
2. 对每一组内的图像做不同尺度的高斯模糊。    

以原始图像为第一层，不断降采样，由此可得到一系列大小不一的图像，由大到小、从下到上构成塔状模型，      
<img src="http://my.csdn.net/uploads/201205/17/1337254665_2720.jpg" />

其中金字塔层数 n 由原始图像尺寸<img src="http://www.forkosh.com/mathtex.cgi? \Medium  (M,N) ">与塔顶图像尺寸 <img src="http://www.forkosh.com/mathtex.cgi? \Medium  (M_t ,N_t) "> 共同决定，如下式：
<img src="http://www.forkosh.com/mathtex.cgi? \Medium  n = {\log _2}\{ \min (M,N)\}  - t,t \in [0,{\log _2}\{ \min ({M_t},{N_t})\} ] ">      *(eq.5)*      

作为尺度连续性的体现，高斯金字塔在简单降采样的基础上加上了高斯滤波。如图3.1所示，将图像金字塔每层内的每张图像使用不同参数做高斯模糊，使得金字塔各层含有多张高斯模糊图像，将每层的多张图像合称为一组(Octave)，金字塔每层只有一组图像，组数和金字塔层数相等，使用公式(3-3)计算，每组含有多张(也叫层Interval)图像。另外，降采样时，高斯金字塔上一组图像的初始图像(底层图像)是由前一组图像的**倒数第三张**图像隔点采样得到的。    

**3.3 高斯差分金字塔**     
2002年Mikolajczyk在详细的实验比较中发现尺度归一化的**高斯拉普拉斯函数** <img src="http://www.forkosh.com/mathtex.cgi? \Medium  {\sigma ^2}{\nabla ^2}G "> 的极大值和极小值较其它特征提取函数（例如梯度、Hessian或Harris角特征比较）能产生最为稳定的图像特征。而Lindeberg早在1994年就发现高斯差分函数（Difference of Gaussian ，简称DOG算子） <img src="http://www.forkosh.com/mathtex.cgi? \Medium  D(x,y,\sigma) "> 与尺度归一化的**高斯拉普拉斯函数**非常近似，二者关系可做如下推导：
<img src="http://www.forkosh.com/mathtex.cgi? \Medium  \frac{{\partial G}}{{\partial \sigma }} = \sigma {\nabla ^2}G ">    
使用差分近似代替微分：    
<img src="http://www.forkosh.com/mathtex.cgi? \Medium  \sigma {\nabla ^2}G = \frac{{\partial G}}{{\partial \sigma }} \approx \frac{{G(x,y,k\sigma ) - G(x,y,\sigma )}}{{k\sigma  - \sigma }} ">    
因此有：    
<img src="http://www.forkosh.com/mathtex.cgi? \Medium  G(x,y,k\sigma ) - G(x,y,\sigma ) \approx (k - 1){\sigma ^2}{\nabla ^2}G ">    
其中常数项 <img src="http://www.forkosh.com/mathtex.cgi? \Medium  (k - 1) "> 不影响极值点位置的求取。如下图所示，红色表示高斯差分DOG算子，蓝色表示高斯拉普拉斯函数。    
<img src="http://my.csdn.net/uploads/201204/28/1335626802_9708.jpg" />       
使用DOG算子近似代替高斯拉普拉斯函数进行极值检测可有效简化计算，提高处理效率。根据公式*(eq.3)*，DOG算子可写为：    
<img src="http://www.forkosh.com/mathtex.cgi? \Medium D(x,y,\sigma ) = (G(x,y,k\sigma ) - G(x,y,\sigma )) \otimes I(x,y) = L(x,y,k\sigma ) - L(x,y,\sigma )">      *(eq.6)*       
实际计算过程中，使用高斯金字塔每组中相邻上下两层图像相减，得到DOG高斯差分图像并进行极值检测，如下图：     
<img src="http://my.csdn.net/uploads/201204/28/1335626876_5968.jpg" />      

**3.4 构建尺度空间所需确定的参数**        
在本文描述的尺度空间中，尺度空间坐标 <img src="http://www.forkosh.com/mathtex.cgi? \Medium \sigma"> ，金字塔组Octave数<img src="http://www.forkosh.com/mathtex.cgi? \Medium O"> 与 组内层数<img src="http://www.forkosh.com/mathtex.cgi? \Medium S"> 的关系如下：

<img src="http://www.forkosh.com/mathtex.cgi? \Medium  \sigma (o,s) = {\sigma_0}{2^{o + \frac{s}{S}}},o \in [0,...,O - 1],s \in [0,...,S + 2] ">       

其中 <img src="http://www.forkosh.com/mathtex.cgi? \Medium  \sigma_0">  为基准层尺度，o为组octave索引，s为组内层索引，关键点的尺度坐标<img src="http://www.forkosh.com/mathtex.cgi? \Medium  \sigma ">即按其所在组与层索引根据上式确定。     
在建立高斯金字塔第一组的基准层时，需对输入图像进行模糊处理，这相当于丢弃了最高层尺度空间的采样率，因此通常做法是先将图像的尺度扩大一倍（例如采用双线性插值）生成第-1组。      
计算组内层数时，取公式*(eq.6)* 中的k值为总层数的倒数：      
<img src="http://www.forkosh.com/mathtex.cgi? \Medium k = {2^{\frac{1}{S}}}">     *(eq.7)*      
则在构建高斯金字塔时，组内每层尺度坐标按下式计算：

<img src="http://www.forkosh.com/mathtex.cgi? \Medium \sigma (s) = \sqrt {{{({k^s}{\sigma_0})}^2} - {{({k^{s - 1}}{\sigma_0})}^2}} ">      *(eq.8)*     

其中初始尺度 <img src="http://www.forkosh.com/mathtex.cgi? \Medium  \sigma0 "> 在Lowe的论文中取1.6，S取3。上式可用于一次计算组内的不同尺度空间坐标。而组内某一层图像的尺度可按下式计算：       
<img src="http://www.forkosh.com/mathtex.cgi? \Medium {\sigma _{oct}}(s) = {\sigma _0}{2^{\frac{s}{S}}},s \in [0,...,S + 2]">        
构建高斯金字塔及DOG金字塔的示意图见下图，原图采用128x128的Jobs图像，扩大一倍后作为基准层构建金字塔。      
<img src="http://my.csdn.net/uploads/201205/17/1337254639_6972.jpg" />      
构建高斯金字塔的实现可参考**OpenCV中SIFT类的`cv::SIFT::buildGaussianPyramid` 方法**，构建DOG金字塔的实现可参考**`cv::SIFT::buildDoGPyramid`** 。    

**3.5 关键点（局部极值点）的初步检测**     
DOG空间的局部极值点组成了所谓的SIFT关键点（Keypoint），关键点的初步探查是通过同一组内各DoG相邻两层图像之间比较完成的。为了寻找DoG函数的极值点，每一个像素点要和它在**尺度空间内所有的相邻点**比较，看其是否比它的图像域和尺度域的相邻点大或者小。如图3.4所示，中间的检测点和它同尺度的8个相邻点和上下相邻尺度对应的9×2个点共26个点比较，以确保在尺度空间和二维图像空间都检测到极值点。     
<img src="http://my.csdn.net/uploads/201204/28/1335626904_5751.jpg" />    
由于要在相邻尺度进行比较，如图3.3右侧每组含4层的高斯差分金子塔，只能在中间两层中进行两个尺度的极值点检测，其它尺度则只能在不同组中进行。为了在每组中检测S个尺度的极值点，则DOG金字塔每组需S+2层图像，而DOG金字塔由高斯金字塔相邻两层相减得到，则高斯金字塔每组需S+3层图像，实际计算时S在3到5之间。    
当然这样产生的极值点并不全都是稳定的特征点，因为某些极值点响应较弱，而且DOG算子会产生较强的边缘响应。      
关键点检测的实现可参考**OpenCV中SIFT类的`cv::SIFT::findScaleSpaceExtrema` 方法**的前半部分（该方法同时还实现关键点插值与方向分配等操作）。         


4.关键点定位
----
以上方法检测到的极值点是离散空间的极值点，还需通过拟合三维二次函数来精确确定关键点的位置和尺度，同时去除低对比度的关键点和不稳定的边缘响应点(因为DoG算子会产生较强的边缘响应)，以增强匹配稳定性、提高抗噪声能力。     

**4.1 关键点的精确定位**     
下图显示了二维函数离散空间得到的极值点与连续空间极值点的差别。利用已知的离散空间点插值（或拟合）得到连续空间极值点的方法叫做子像素插值（Sub-pixel Interpolation）。     
<img src="http://my.csdn.net/uploads/201204/28/1335627281_8397.jpg" />     
为提高关键点的数学稳定性，需对尺度空间中的DOG函数进行拟合，其在尺度空间的泰勒展开式如下：     
<img src="http://www.forkosh.com/mathtex.cgi? \Medium D(X) = D + \frac{{\partial {D^T}}}{{\partial X}}X + \frac{1}{2}{X^T}\frac{{{\partial ^2}D}}{{\partial {X^2}}}X">      
其中 <img src="http://www.forkosh.com/mathtex.cgi? \Medium  X = {(x,y,\sigma )^T}"> 求导并使方程为零，可得到极值点偏移量：    
<img src="http://www.forkosh.com/mathtex.cgi? \Medium  \hat X =  - \frac{{{\partial ^2}{D^{ - 1}}}}{{\partial {X^2}}}\frac{{\partial D}}{{\partial X}}">      
方程的求导方法实现可参考OpenCV相关代码。      
当偏移量在任一维度上的值大于0.5则表示插值中心已偏移至邻近点，需改变当前关键点的位置，同时在新位置上继续拟合至收敛；新位置可能超出图像边界或在设定的迭代次数内未收敛，则此时应删除该点。另外，<img src="http://www.forkosh.com/mathtex.cgi? \Medium  \left| {D(X)} \right|"> 过小的点易受噪声干扰，因此可将 <img src="http://www.forkosh.com/mathtex.cgi? \Medium  \left| {D(X)} \right|"> 小于某个经验值（Lowe的论文中使用0.03，Rob Hess的实现使用0.04/S）的极值点删除。
最终，在上述过程中可求解得到关键点的精确位置及尺度。实现代码可参考**OpenCV中SIFT类的`cv::SIFT::findScaleSpaceExtrema`方法**。     
     
 
**4.2 消除边缘响应**     
DOG算子内在会产生较强的边缘响应，实际应用时需剔除不稳定的边缘响应点。一个平坦的DoG响应峰值在横跨边缘的地方有较大的主曲率，而在垂直边缘的地方有较小的主曲率。主曲率可以通过2×2的Hessian矩阵H求出：      

<img src="http://www.forkosh.com/mathtex.cgi? \Medium H(x,y) = \left( {\begin{array}{*{20}{c}}
{{D_{xx}}(x,y)}&{{D_{xy}}(x,y)}\\
{{D_{xy}}(x,y)}&{{D_{yy}}(x,y)}
\end{array}} \right) ">    


D值可以通过求临近点差分得到。H的特征值与D的主曲率成正比，具体可参见Harris角点检测算法。
为了避免求具体的值，我们可以通过H将特征值的比例表示出来。令  <img src="http://www.forkosh.com/mathtex.cgi? \Medium  \alpha=\lambda_{max} ,\beta={\lambda_{\min}} "> 分别为最大和最小特征值，则矩阵H的迹和行列式可表示如下：     
<img src="http://www.forkosh.com/mathtex.cgi? \Medium  Tr(H) = {D_{xx}} + {D_{yy}} = \alpha  + \beta \\
Det(H) = {D_{xx}} \cdot {D_{yy}} = \alpha \cdot \beta ">        
令<img src="http://www.forkosh.com/mathtex.cgi? \Medium  \gamma  = \alpha /\beta "> 表示最大与最小特征值之比，则有：     
<img src="http://www.forkosh.com/mathtex.cgi? \Medium  \frac{{Tr{{(H)}^2}}}{{Det(H)}} = \frac{{{{(\alpha  + \beta )}^2}}}{{\alpha  \cdot \beta }} = \frac{{{{(\gamma  + 1)}^2}}}{\gamma }">      
上式与两个特征值的比例有关，其值越大，说明两个特征值的比值越大，即在某一个方向的梯度值越大，而在另一个方向的梯度值越小，而边缘恰恰就是这种情况。随着主曲率比值的增加，<img src="http://www.forkosh.com/mathtex.cgi? \Medium  \frac{{{{(\gamma  + 1)}^2}}}{\gamma }"> 也会增加，因此剔除边缘响应时只要去掉比率大于一定值的关键点即可。Lowe论文中取比值=10。       
消除边缘响应的代码可参考**OpenCV的`cv::adjustLocalExtrema` 方法** 。    
    

5.关键点方向分配
----       
为了使特征描述符具有旋转不变性，需要利用图像的局部特征为给每一个关键点分配一个基准方向。Lowe论文中使用图像梯度的方法求取局部结构的稳定方向。
**5.1 梯度方向和幅值**     
对于在DOG金字塔中检测出的关键点，使用有限差分方法，采集其所在高斯金字塔图像<img src="http://www.forkosh.com/mathtex.cgi? \Medium  3\sigma ">邻域窗口内像素的梯度和方向分布特征。梯度的模值和方向如下：     
<img src="http://www.forkosh.com/mathtex.cgi? \Medium m(x,y) = \sqrt {{{(L(x + 1,y) - L(x - 1,y))}^2} + {{(L(x,y + 1) - L(x,y - 1))}^2}} ">      
<img src="http://www.forkosh.com/mathtex.cgi? \Medium  \theta (x,y) = {\tan ^{ - 1}}\left( {\frac{{L(x,y + 1) - L(x,y - 1)}}{{L(x + 1,y) - L(x - 1,y)}}} \right)">       
其中 <img src="http://www.forkosh.com/mathtex.cgi? \Medium L"> 是关键点所在尺度的高斯模糊图像。    

**5.2 梯度直方图**     
在完成关键点邻域内高斯图像梯度计算后，使用直方图统计邻域内像素对应的梯度方向和幅值。有关直方图的基础知识可以参考[《数字图像直方图》](http://blog.csdn.net/xiaowei_cqu/article/details/7600666)，可以看做是离散点的概率表示形式。此处方向直方图的核心是统计以关键点为原点，一定区域内的图像像素点对关键点方向生成所作的贡献。   
梯度方向直方图的横轴是梯度方向角，纵轴是剃度方向角对应的梯度幅值累加值。梯度方向直方图将0°~360°的范围分为36个柱，每10°为一个柱。下图是从高斯图像上求取梯度，再由梯度得到梯度方向直方图的例图。     
<img src="http://img.my.csdn.net/uploads/201210/21/1350823039_5697.png" />    
在计算直方图时，每个加入直方图的采样点都使用圆形高斯函数函数进行了加权处理，也就是进行高斯平滑。这主要是因为SIFT算法只考虑了尺度和旋转不变形，没有考虑仿射不变性。通过高斯平滑，可以使关键点附近的梯度幅值有较大权重，从而部分弥补没考虑仿射不变形产生的特征点不稳定。     
通常离散的梯度直方图要进行插值拟合处理，以求取更精确的方向角度值。（同"关键点搜索与定位"中插值拟合的思路）    
获得图像关键点主方向后，每个关键点有三个信息(x,y,σ,θ)，即：二维坐标、尺度、方向。由此我们可以确定一个SIFT特征区域。通常使用一个带箭头的圆或直接使用箭头表示SIFT区域的三个值：中心表示特征点位置，半径表示关键点尺度（r=2.5σ）,箭头表示主方向。具有多个方向的关键点可以复制成多份，然后将方向值分别赋给复制后的关键点。如下图：     
<img src="http://img.my.csdn.net/uploads/201210/21/1350826032_9513.png" />    
实现方式可参考**OpenCV的 `cv::calcOrientationHist` 方法**，该方法在**`cv::findScaleSpaceExtrema` **中被调用。
至此，检测出的含有位置、尺度和方向的关键点即是该图像的**SIFT特征点**。      

6. 关键点特征描述     
----   
接下来的步骤是关键点描述，即用用一组向量将这个关键点描述出来，这个描述子不但包括关键点，也包括关键点周围对其有贡献的像素点。用来作为目标匹配的依据（所以描述子应该有较高的独特性，以保证匹配率），也可使关键点具有更多的不变特性，如光照变化、3D视点变化等。      
SIFT描述子h(x,y,θ)是对关键点附近邻域内高斯图像梯度统计的结果，是一个三维矩阵，但通常用一个矢量来表示。矢量通过对三维矩阵按一定规律排列得到。Lowe建议描述子使用在关键点尺度空间内4x4的窗口中计算的8个方向的梯度信息，共 <img src="http://www.forkosh.com/mathtex.cgi? \Medium  4\times4\times8=128 "> 维向量表征。      
**6.1 描述子采样区域**      
特征描述子与关键点所在尺度有关，因此对梯度的求取应在特征点对应的高斯图像上进行。将关键点附近划分成d×d个子区域，每个子区域尺寸为mσ个像元（根据Lowe论文，取d=4，m=3，σ为尺特征点的尺度）。考虑到实际计算时需要双线性插值，故计算的图像区域为mσ(d+1)，再考虑旋转，则实际计算的图像区域为<img src="http://www.forkosh.com/mathtex.cgi? \Medium m\sigma (d + 1)\sqrt 2 "> ，如下图：    
<img src="http://img.my.csdn.net/uploads/201210/26/1351213289_3302.png" />    
**6.2 区域坐标轴旋转**      
为了保证特征矢量具有旋转不变性，要以特征点为中心，在附近邻域内旋转θ角，即旋转为特征点的方向。    
<img src="http://img.my.csdn.net/uploads/201210/26/1351213506_4881.png" />    
旋转后区域内采样点新的坐标为：    
<img src="http://www.forkosh.com/mathtex.cgi? \Medium \left( \begin{array}{l}
{x'}\\
{y'}
\end{array} \right) = \left( {\begin{array}{*{20}{c}}
{\cos \theta }&{ - \sin \theta }\\
{\sin \theta }&{\cos \theta }
\end{array}} \right)\left( {\begin{array}{*{20}{c}}
x\\
y
\end{array}} \right)">  
将旋转后区域划分为d×d个子区域（每个区域间隔为mσ像元），在子区域内计算8个方向的梯度直方图，绘制每个方向梯度方向的累加值，形成一个种子点。     
与求主方向不同的是，此时，每个子区域梯度方向直方图将0°~360°划分为8个方向区间，每个区间为45°。即每个种子点有8个方向区间的梯度强度信息。由于存在d×d，即4×4个子区域，所以最终共有4×4×8=128个数据，形成128维SIFT特征矢量。     
<img src="http://img.my.csdn.net/uploads/201210/26/1351217343_1515.png" />     
对特征矢量需要加权处理，加权采用mσd/2的标准高斯函数。为了除去光照变化影响，还有加上归一化处理。
以上处理步骤的实现可参考**OpenCV的 `cv::calcSIFTDescriptor` 方法** 
至此描述子生成，SIFT算法基本完成。
