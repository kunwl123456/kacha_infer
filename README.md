## 第三方依赖
> 借助企业级开发库，更快地搭建出大模型推理框架
1. google glog https://github.com/google/glog
2. google gtest https://github.com/google/googletest
3. sentencepiece https://github.com/google/sentencepiece
4. armadillo + openblas https://arma.sourceforge.net/download.html
5. Cuda Toolkit


## 模型下载地址
1. LLama2 https://pan.baidu.com/s/1PF5KqvIvNFR8yDIY1HmTYA?pwd=ma8r 或 https://huggingface.co/fushenshen/lession_model/tree/main

2. Tiny LLama 
- TinyLLama模型 https://huggingface.co/karpathy/tinyllamas/tree/main
- TinyLLama分词器 https://huggingface.co/yahma/llama-7b-hf/blob/main/tokenizer.model

3. Qwen2.5/LLama
4. Qwen3.0

## 版本更新

### 1.10/08/2025
1)增加内存和显存管理
2)增加测试内存和显存用例目录

## 2.10/09/2025
1)增加矩阵运算测试用例
2)增加tensor相关类
3)增加tensor 测试用例

## 3.10/20/2025
1)完善llm tensor相关代码

## 4.10/22/2025
1)RMSNorm算子的CUDA实现
RMSNorm算子的计算公式:
其中d是输入的x向量的维度，w是权重需要在计算1，2两式完成时对原结果进行逐点相乘。x是算子计算的输入，y是算子计算的输出。
https://l0kzvikuq0w.feishu.cn/docx/BXtyd0xGHoYWFgxrPDkcw2HXnxb#share-LL6PdBycXo9KALxMIl6cowksnce
2)RMSNorm算子测试样例
案例1	480	1D	❌	基础功能
案例2	4×1024	2D	❌	多维 + in-place
案例3	32	1D	✅	异步执行
案例4	72,480	1D	✅	大规模数据

## 5.11/05/2025
1)RMSNorm算子优化：(从warpreduce到blockreduce)
需要优化的点：总是以32个线程为规约(warpSize)，但是一个block内有64个线程那么tid的范围就是0到63，lane_id范围就是0到31(跟随warpSize改变)，而tid等于32到64的时候，lane_id取值范围仍然为0到31（总结：工作的只有32个线程）
const int lane_id = tid % warpsize
解决方法：以block之间规约，假设一个block有128个线程，需要处理数据是1024个，每个线程需要处理8个数据，因为1024/128=8,假设每个数值均为1
（1）线程0需要负责将下标为0、128、256...位置上共8个数据求和
（2）线程1需要负责将下标1、129、257...位置上共8个数据的求和，以此类推每个线程的求和结果均为8
（3）对128个线程没32个进行一次reduce，每32个线程也就是以一个warp作为单位进行规约。将值存到共享内存中sh_mem中
线程32-63做一次规约，将值存到共享内存中下一个位置中。
线程64到95做一次规约，将值存到共享内存的再下一个位置中。
（4）经过上一步骤，在共享内存中sh_mem数组中，第0个位置存放线程0-31的求和结果，也就是256
第1个位置存放线程32-63的求和结果，同样是256
第2个坐标位置存放线程64-95的求和结果，同样是256
第3个坐标位置存放线程96-127的求和结果。现在一共得到了4个局部和，也就是每个warp各自的和
（5）我们需要对存储在共享内存 sh_mem 中的局部求和结果进行进一步的规约求和操作，以得到全局总和（即对这四个局部求和结果进行规约）。具体来说，就是将这四个局部和数值相加，从而得出最终的全局总和。
