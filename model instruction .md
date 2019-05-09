# Baikal 交接文档-模型部分


## 主要功能

针对多元数据，提供了基于polygon的时间序列提取，预处理以及模型训练功能，设计多种模型的指令调用操作。文档将包括模型调用指令说明。


---

### 模型调用说明

- 涉及以下几种模型： 随机森林（DT），全连接神经网络（MLP），SVM，以及LSTM模型。
- 模型文件路径： /model/
- 运行主程序： /script/run-model.py
- 输入文件： 

    LSTM文件输入格式参考 /data2/citrus/demo/sample_result/TQLS/sichuan/arfar-megatron_48RVU_tqls_20190319T114813/TD_S3_L3a_20190319T114820_extract.npz
    
    其余模型输入文件参考 /data2/citrus/demo/sample_result/TQLS/sichuan/arfar-megatron_48RVU_tqls_20190319T114813/TD_S3_L3a_20190319T114820_TRAIN.npz

- 程序使用说明：

    输入 python run-model.py -h 获取帮助信息， 显示如下：
        
        Model training. Use -h for more information.

        optional arguments:
        -h, --help            show this help message and exit

        subcommands:
        Command for local training.

        {use-rnn,ur,use-mlp,um,use-svm,us,use-dt,ud}
            use-rnn (ur)        Command for Long-Short term Memory model train.
            use-mlp (um)        Command for MLP model train.
            use-svm (us)        Command for SVM model train.
            use-dt (ud)         Command for Decision Tree model train.

    输入 python run-model.py ur -h 获取LSTM模型调用的指令帮助说明，显示如下：

        usage: run-model.py use-rnn [-h] [-model-type MODEL_TYPE]
                            [-start-date START_DATE] [-end-date END_DATE]
                            [-sg-param SG_PARAM] [-chunk-size CHUNK_SIZE]
                            [-learning-rate-init LEARNING_RATE_INIT]
                            [-num-classes NUM_CLASSES] [-max-iter MAX_ITER]
                            [-dropout DROPOUT]
                            [-num-hidden-units NUM_HIDDEN_UNITS]
                            [-num-layers NUM_LAYERS]
                            [-learning-decay LEARNING_DECAY]
                            target-dir data-file batch-name lstm-state
        ...(请在命令行运行获取更多信息，包含各参数说明)
        其中，LSTM涉及数据预处理过程，参数相对复杂，请认真阅读说明。

    输入 python run-model.py um -h 获取MLP模型调用的指令帮助说明，显示如下：

        usage: run-model.py use-mlp [-h] [-model-type MODEL_TYPE]
                            [-hidden-layer-sizes HIDDEN_LAYER_SIZES]
                            [-learning-rate-init LEARNING_RATE_INIT]
                            [-max_iter MAX_ITER] [-test-size TEST_SIZE]
                            target-dir data-file batch-name
        ...(请在命令行运行获取更多信息，包含各参数说明)

    输入 python run-model.py us -h 获取SVM模型调用的指令帮助说明，显示如下：

        usage: run-model.py use-svm [-h] [-model-type MODEL_TYPE] [-gamma GAMMA]
                            [-C C] [-kernel KERNEL]
                            target-dir data-file batch-name
        ...(请在命令行运行获取更多信息，包含各参数说明)

    输入 python run-model.py ud -h 获取DT模型调用的指令帮助说明，显示如下：

        usage: run-model.py use-dt [-h] [-model-type MODEL_TYPE]
                           [-max-depth MAX_DEPTH] [-criterion CRITERION]
                           target-dir data-file batch-name
        ...(请在命令行运行获取更多信息，包含各参数说明)

---
以上

-- U，2019/5/8



