# Time-Series-Prediction-System

## 项目介绍
**本项目是一个支持机器学习、深度学习和集成学习的多因素时间序列预测系统，旨在能够根据历史数据和外部特征对时间序列进行多步预测。**

- 本项目提供了丰富的接口和简便的实现方法，特别是针对深度学习模型的构建，用户只需提供模型类就可以自动执行数据封装、模型训练、模型评估、预测绘图和保存等功能。
- 本项目已经内置了 sklearn 常用回归模型（SVR、Ridge、RandomForestRegressor、GradientBoostingRegressor、AdaBoostRegressor、BaggingRegressor）、多层感知机（MLP）、循环神经网络系列模型（RNN、LSTM、GRU）和 Transformer系列模型（TranMLP、TranAttnFNN）。用户可以提供自定义的模型类（序列模型需继承自 BaseSequence 类、回归模型需继承自 BaseRegression），无需考虑训练、评估、展示和保存等繁杂工作，该项目可以自动实现以上功能（**仅支持来自 sklearn 和 torch 的模型**）。
- 本项目提供了 Bagging 和 Stacking 集成学习，用户可以通过传入基学习器输出结果的路径就可以将预测结果进一步集成，得到更稳定的预测结果。用户仅需修改 data.yaml 中的 EnsembleMethod 参数即可快速实现集成，集成过程支持所有模型，同时可以自动执行训练、评估、绘图、保存等功能。
- 本项目支持多步预测任务，并且用户可以为不同的预测步设置不同的损失和评价权重。本项目将外部特征划分为时变已知特征和时变未知特征，并支持特征选择和模型选择，以实现对不同类型的特征进行精细化处理。本项目支持对训练集按照指定间隔进行下采样，以降低过拟合风险。本项目支持 K 折交叉验证，既能用于评估模型，又能用于后续的模型集成。
- 本项目具有良好的封装结构，用户仅需修改 yaml 配置文件，在 main.py 中导入自定义的配置文件即可实现所有内置功能。用户可以自定义模型参数、训练配置、学习率调度器，以及是否输出信息到控制台和保存哪些文件，具有较高的可拓展性和个性化设计。

## 环境要求（仅供参考）
Python = 3.10 </br>
numpy = 1.26.3 </br>
pandas = 2.2.3 </br>
matplotlib = 3.10.0 </br>
scikit-learn = 1.6.1 </br>
torch = 2.5.1+cu124 </br>
tqdm = 4.67.1 </br>

## 参数设置
- 所有参数文件（yaml 文件）均存放在 parameters 目录下，包含 data.yaml 和 model.yaml 文件。
- data.yaml 文件中存放数据封装和保存路径等参数。例如：是否启用 Stacking 集成学习、保存参数、数据集封装参数、数据集采样参数；对于基学习器有数据集划分比例和数据集特征参数两个专属参数；对于集成学习有模型选择专属参数。
- model.yaml 文件中存放各类模型参数和训练参数。例如：仅支持单步预测的回归模型（SVR、GradientBoostingRegressor、AdaBoostRegressor）；支持多步预测的回归模型（Ridge、RandomForestRegressor、BaggingRegressor、MLP、CAttn、CAttnProj）；支持多步预测的时序模型（RNN、LSTM、GRU、TranMLP、TranAttnFNN）。

## 开始运行
1. 在项目根目录创建 **datasets** 文件夹，并将数据文件放入该文件夹中，支持 csv、xlsx、xls 文件类型
2. 修改 **data.yaml** 和 **model.yaml** 中的数据和模型参数（可选择是否启用 Stacking 集成学习，如果启用 Stacking 集成学习需保证 ReadResultDir 传入的为基学习器输出目录，如果不启用集成学习需保证 FileName 传入的为 datasets 目录下存在的文件名）。
3. 修改 **main.py** 中 data_parameter_path 和 model_parameter_path 变量，读取指定的参数文件。
4. 修改 **main.py** 中的 criterion 变量，选择合适的损失函数（例如：Huber_loss 适用于点预测，QuantileLoss 适用于分位数预测）。注：QuantileLoss 的分位数参数请修改 utils.criterion 中的 QuantileLoss.tau 值。
5. 修改 **main.py** 中的 monitor 变量，选择合适监视器函数（所有的损失函数和监视器函数均需要来自于 utils.criterion 中，也可以自定义）
6. 修改 **main.py** 中的 models 和 normalizations 变量，选择使用的预测模型和是否标准化
7. 在 **main.py** 中初始化 **Runer** 类，并通过 **Runer.run** 方法启动预测系统。

Note: 
- 参数文件中的每个参数含义在对应位置上已添加注释。
- 运行案例参考 main.py 文件。

## 预测结果
生成的结果将会保存到 **result** 目录下，保存的根目录由 **data.yaml** 中的 ***SaveResultDir*** 参数决定，以及用户也可以选择当保存的根目录存在时是否删除该目录。该程序将会自动在保存的根目录下生成 **"data", "documents", "figures", "models", "results" 和 ".identify"**，其中：
- **"data"** 保存数据封装后的特征和目标对应关系（.xlsx），以及可以直接用于机器学习和深度学习的的变量（.pkl）。
- **"documents"** 保存日志信息和用户设置的参数信息（如果运行出现错误也会保存错误信息）。
- **"figures"** 保存预测走势图、深度学习损失函数图和监控函数图。
- **"models"** 保存训练后的模型，对于深度学习模型会保存在验证集效果最好的模型以及训练结束的模型。来自 sklearn 的机器学习模型保存格式为 **.pkl**，来自 pytorch 的深度学习模型保存格式为 **.pth**。
- **"results"** 保存预测结果、评估指标结果，如果是深度学习模型还会保存损失函数值、监控器值、学习率和训练每秒样本数。如果是多步预测预测不仅会保存总体评估指标，也会保存每一步的预测和评估指标。总体评估指标的计算受 **data.yaml** 中 ***Weights*** 参数影响。
- **".identify"** 用于识别当前目录是否由该系统创建，用于识别预测结果和删除保护。

Note: 在深度学习模型中，最终用于预测和评估的模型不是最后一次迭代的模型，而是整个训练周期中，验证集在监控函数中表现最好（监控器取最低值）的模型。

## 项目模块介绍
如果您想深度适配您的数据或工作，您可以阅读下面的内容，以便您更好的了解整个项目的逻辑。

### 基础模块（base.base.py）
- **BaseModel** 类是所有自定义深度模型的基类（用于预测），它内部定义了 **fit**（深度学习模型训练）、**evaluate**（深度学习模型评估）、**to_device**（模型转移到指定设备）、**save**（保存模型）方法。在子类中需要重写 **\_\_init\_\_**、**forward**、**train_epoch**（训练一轮）和 **predict**（深度学习模型推理）方法。
- **BaseSequence** 类是所有深度学习时序模型的基类（例如：RNN等），它继承自 BaseModel 类，重写了 train_epoch 和 predict 方法。该子类接受的输入维度为：[batch_size, time_step, input_size]，输出维度为：[batch_size, output_size]。
- **BaseRegression** 类是所有深度学习回归模型的基类（例如：MLP等），它继承自 BaseModel 类，重写了 train_epoch 和 predict 方法。该子类接受的输入维度为：[batch_size, input_size]，输出维度为：[batch_size, output_size]。
- 不建议自定义的模型继承自 BaseModel 基类，建议继承自 BaseSequence 或 BaseRegression 子类，即使需要改写 train_epoch 和 predict 也建议根据输入的维度继承自合适的子类。
- 自定义的模型需要重写 \_\_init\_\_ 和 forward 方法，重写的要求与继承自 nn.Module 一致。
- **BaseModel** 类中的 **fit** 方法会**自动训练深度学习模型**、**保存最优模型**、**控制台打印并保存损失值和监测值**、**绘制动态损失值图像和监测值图像**、以及**保存所有训练结果**等操作（这些过程用户都可以通过传入参数自行决定是否执行）。

### 数据封装（data）
- **read.py** 实现了数据文件的读取（基学习器）、预测结果的读取（集成学习）和数据封装。其中，**ReadDataset** 类处理数据集文件，并将其转换为机器学习或深度学习数据格式用于基学习器。**ReadResult** 类读取基学习器的预测结果和特征值，并处理成指定的机器学习或深度学习数据格式用于集成学习。**PackageDataset** 类将读取的数据封装成可用于训练的格式（包括 ndarray 和 DataLoader 等），并划分数据集、检查数据是否对齐和训练数据写入本地。**ReadParameter** 类读取配置文件的参数和写入参数。
- **selector.py** 中的 **Selector** 模块实现了特征选择和模型选择，既支持使用统计方法进行选择，也支持使用评价指标进行选择（仅限集成学习）。
- **scaler.py** 中的 **Norm** 模块实现对特征和目标的标准化和反标准化（正态），也可以传入新的数据进行标准化和反标准化。

### 回归预测（regression）
- **attention.py** 实现了 CAttention、CAttn、CAttnProj 和 AttnProj 四个基于注意力机制的回归模型。
- **mlp.py** 实现了 MLP （多层感知机）回归模型。
- **cnn.py** 实现了基于卷积神经网络的 C2L 回归模型。

### 时序预测（sequence）
- **transformer.py** 实现了 TranMLP 和 TranAttnFNN 两个基于 Transformer 的时序预测模型。
- **rnn.py** 基于 BaseSequence 实现了 BaseRNN 子类，用于实现循环神经网络系列模型。通过继承 BaseRNN 实现 RNN、LSTM 和 GRU 三个循环神经网络模型。

### 工具模块（utils）
- **criterion.py** 实现了 Huber 损失/评估函数（Huber_loss）、均方误差损失/评估函数（MSELoss）、平均绝对误差损失/评估函数（MAELoss）、对称平均绝对百分比误差损失/评估函数（sMAPELoss）、平均绝对百分比误差损失/评估函数（MAPELoss）、分位数损失/评估函数（QuantileLoss）。用户可以根据需要选择合适的损失函数和监控器函数，也可以自定义新的损失函数和监控器函数。
- **writer.py** 文件中的 Writer 类可以暂存和保存表数据、文本数据和文件数据，也可以用于保存机器学习和深度学习模型，还可以绘制和保存图像。Writer 类支持删除上次输出结果，也可以在上次输出结果中继续追加新数据。
- **metrics.py** 文件中的 **calculate_metrics** 函数实现了多种不同的点评估指标，已经内置了 **MSE、RMSE、MAE、MAPE、SMAPE、MSLE、RMSLE、MedAE、MaxError、RAE、RSE、Huber、R2** 指标。默认输出所有指标。
- **timer.py** 文件中的 Timer 类用于记录时间。
- **device.py** 文件中的 try_gpu 和 try_all_gpus 函数分别返回指定的一个 GPU 设备和返回所有可用的 GPU 设备。
- **cprint.py** 文件中的 cprint 函数是基于 ANSI 转义序列实现的控制台彩色打印和彩色背景。
- **animator.py** 文件中的 Animator 类是一个动态绘图类。
- **accumulator.py** 文件中的 Accumulator 类是一个累加器。

### 扩展模块（extensions）
- **probabilistic_metrics.py** 文件中的 calculate_possibility 函数实现了计算概率预测评价指标； calculate_possibility_folds 函数实现了从指定的目录中读取预测结果，并且计算概率预测评价指标。支持预测区间覆盖率（PICP）、平均区间宽度（MPIW）、区间归一化平均宽度（PINAW）、区间分数（IS）、连续排名概率评分（CRPS）、中心点绝对误差（MAE_MidPoint）、中心点均方误差（MSE_MidPoint）和覆盖宽度标准（CWC）。

### 其他文件
- **runner.py** 为预测模型和 Stacking 集成的封装程序。主要内容包括：将原始数据封装为机器学习和深度学习格式的数据集、训练预测模型或集成模型、保存日志、将所有暂存内容全部写入本地文件。（程序运行过程中如果遇到任何继承于 Exception 的错误，都将会保存错误至 **documents\\Logs.log** 下，然后将所有暂存的结果全部保存到本地，防止数据丢失）。
- **main.py** 为运行主程序。用户选择运行预测器还是集成器、选择需要训练的模型和是否对模型进行数据标准化操作。
