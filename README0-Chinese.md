# 基于人体生理数据的睡眠分层研究
# 毕业设计
## 1. 研究背景 Introduction
睡眠分期是研究睡眠及其相关疾病的基础，是完成睡眠质量评估的前提。近年来，基于计算机技术的脑电信号自动分期成为了研究热点。本研究基于睡眠过程中的脑电图(EEG)生理信号数据，采用基于特征工程的机器学习算法和深度神经网络算法训练模型，从而实现睡眠的自动分期。

睡眠是一种复杂的生理过程，是机体复原和巩固的重要环节，对个体的表现、学习能力和身体机能起着至关重要的作用[1-9]。随着现代人生活压力越来越大，许多心血管疾病和精神疾病都与睡眠有关。如文献[6][20]表明，90%以上的抑郁症患者患有睡眠障碍。而睡眠分期是研究睡眠及相关疾病的基础，是完成睡眠质量评估的前提。目前，睡眠分期方法包括两种，一种为人工手动判别，一种为计算机辅助自动分期。人工手段判需要相关领域的专家依靠视觉来进行分析，不仅效率低下，而且容易造成误判。计算机辅助分期则是利用现代信号处理技术对睡眠进行自动分期，高效、客观，是现代睡眠分期研究的主要方法[Hamida S T B, Ahmed B. Computer based sleep staging: challenges for the future[C]//2013 7th IEEE GCC Conference and Exhibition (GCC). IEEE, 2013: 280-285.]。睡眠脑电图（elecroencephalogram, EEG）是睡眠各个阶段的生理信号记录，包含着许多重要的生理病理信息。使用EEG信号进行分析和研究睡眠有利于客观评估睡眠质量，以及预防和诊断睡眠相关疾病。

本课题基于生物医学信号研究网站PhysioNet的Sleep-EDFx数据集，从睡眠信号中习得睡眠生物信号特征与睡眠时期之间的映射关系，进而构建出睡眠分期的深度学习模型。

## 2. 数据集（DataSet）
本研究所采用的数据集为开源医学生物信号研究网站[PhysioNet](www.physionet.org)在.目前暂定使用两组数据：

|数据集名称|下载网址|描述|备注|
|----|----|----|----|
|PhysioNet Challenge 2018|https://physionet.org/content/challenge-2018/1.0.0/|采样率200 Hz|很大，大概200多G|
|Sleep-EDFx|https://physionet.org/content/sleep-edfx/1.0.0/|采样率100 Hz|已下载|

### SleepNetEdx
数据包涵197个人的整晚生理信号数据，包括EEG(Electroencephalogram,脑电图)，EOG（electro-oculogram，眼电图），chin EMG(颏肌电图)等数据。这些数据都被专业的人士通过手动进行过睡眠分层。每个人的数据已经经过专业人士的处理。睡眠的模式被分成W, R, 1, 2, 3, 4, M(Movement time)和?(not scored).
所有EOG，EMG和EEG信号的采样率都为**100Hz**.

### PhysioNet Challenge 2018数据

数据已经有专业的人士处理过，被分为六个阶段，分别为W, S1, S2, S3, REM和未知。所有的信号都为**200Hz**采样率。

## 3. 预期实验算法安排（Methods & Experiment)
### 3.1 支持向量机(SVM)

### 3.2 卷积神经网络(CNN)

### 3.3 循环神经网络（RNN）

### 3.4 睡眠评价(Sleep Scoring)——CNN与LSTM



