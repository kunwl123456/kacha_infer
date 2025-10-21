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

## 4.10/21/2025
1)RMSNorm算子的CUDA实现
RMSNorm算子的计算公式:
其中d是输入的x向量的维度，w是权重需要在计算1，2两式完成时对原结果进行逐点相乘。x是算子计算的输入，y是算子计算的输出。
https://l0kzvikuq0w.feishu.cn/docx/BXtyd0xGHoYWFgxrPDkcw2HXnxb#share-LL6PdBycXo9KALxMIl6cowksnce
2)RMSNorm算子测试样例
案例1	480	1D	❌	基础功能
案例2	4×1024	2D	❌	多维 + in-place
案例3	32	1D	✅	异步执行
案例4	72,480	1D	✅	大规模数据