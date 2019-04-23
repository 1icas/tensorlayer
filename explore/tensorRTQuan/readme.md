量化思路来源： http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

目前在一个cifar-10网络上成功实现了量化。 量化前准确率0.8340 量化后准确率0.8343

当前文件夹中的model.npz为float32模型文件，model1.npz为量化后文件