#include <device_launch_parameters.h>
#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"


namespace kernel
{
    static __global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size,int size, float eps) 
    {
        const int bid = blockIdx.x;
        const int tid = threadIdx.x;
        //// 如果 Block ID 超出范围
        if (bid >= dim_size) {
            return;
        }

        //计算当前 Block 要处理的输入数据的起始地址,bid:block ID,size:每行的数据
        float* block_in = in + bid * size;
        float* block_out = out + bid * size;
        constexpr int pack_size = 4;
        const int pack_num = size / pack_size;
        const int pack_off = pack_size * pack_num;

        float sum = 0.0f;
        float4* in_pack = reinterpret_cast<float4*>(block_in);
        for (int i = tid; i < pack_num; i += blockDim.x) {
            float4 in_float4 = *(in_pack + i);
            sum += in_float4.x * in_float4.x;
            sum += in_float4.y * in_float4.y;
            sum += in_float4.z * in_float4.z;
            sum += in_float4.w * in_float4.w;
        }

        for (int i = pack_off + tid; i < size; i += blockDim.x) {
            sum += block_in[i] * block_in[i];
        }

        using BlockReduce = cub::BlockReduce<float, 128>;
        __shared__ typename BlockReduce::TempStorage temp;
        __shared__ float shared_val;
        sum = BlockReduce(temp).Sum(sum);
        if (threadIdx.x == 0) {
            shared_val = sum;
        }
        __syncthreads();
        sum = shared_val;
        const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

        float4* wei_pack = reinterpret_cast<float4*>(wei);
        float4* out_pack = reinterpret_cast<float4*>(block_out);
        for (int i = tid; i < pack_num; i += blockDim.x) {
            float4 in_float4 = *(in_pack + i);
            float4 wei_float4 = *(wei_pack + i);
            *(out_pack + i) =
                make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                            scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
        }

        for (int i = pack_off + tid; i < size; i += blockDim.x) {
            block_out[i] = wei[i] * block_in[i] * scale;
        }
    }


    //这里BLOCK_DIM是模板参数，传进来是128
    template <int32_t BLOCK_DIM>
    static __global__ void row_rmsnorm_f32(float* in,float* wei,float* out ,int size,float eps)
    {
        //tid：当前线程在 Block 内的索引
        const int tid = threadIdx.x;
        //每次读取 4 个 float（使用 float4 类型），一次读取 4×4=16 字节，提高内存带宽利用率
        constexpr int pack_size = 4;
        //总共可以打包多少组,例如：size = 4096, pack_num = 4096 / 4 = 1024 组
        const int pack_num = size /pack_size;
        //打包处理完后的起始位置（剩余元素的起点）,用于表示处理不能被4整除的显存地址
        const int pack_off = pack_size * pack_num;

        float sum = 0.0f;
        float4* in_pack = reinterpret_cast<float4*>(int);
        //将 float* 指针转换为 float4* 指针,这样可以一次读取 4 个连续的 float
        for(int i = tid;i< pack_num;i+=blockDim.x)
        {
            float4 in_float4 = *(in_pacl + i);
            sum += in_float4.x * in_float4.x;
            sum += in_float4.y * in_float4.y;
            sum += in_float4.z * in_float4.z;
            sum += in_float4.w * in_float4.w;
        }



        __shared__ typename BlockReduce::TempStorage temp;
        __shared__ float shared_val;
        //使用树形归约算法，时间复杂度 O(log N),将所有线程的 sum 相加
        sum = BlockReduce(temp).Sum(sum);
        //只有线程 0 的 sum 会存储最终结果，其他线程的 sum 值未定义
        if(threadIdx.x == 0)
        {
            shared_val = sum;
        }
        __syncthreads();
        sum = shared_val;


        //scale就是开平方分之一（rsqrt = reciprocal sqrt = 平方根的倒数），cuda内置函数
        const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
        float4* wei_pack = reinterpret_cast<float4*>(wei);
        float4* out_pack = reinterpret_cast<float4*>(out);
        //使用 float4 每次写 4 个元素
        for (int i = tid; i < pack_num; i += blockDim.x) {
            float4 in_float4 = *(in_pack + i);
            float4 wei_float4 = *(wei_pack + i);
            *(out_pack + i) = make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
        }
        //再写入不能被4整除的剩余数据
        for(int i = pack_off +tid; i<size;i+= blockDim.x)
        {
            block_out[i] = wei[i] * block_in[i] * scale;
        }

    } 

    void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream)
    {
        CHECK(!input.is_empty());
        CHECK(!weight.is_empty());
        CHECK(!output.is_empty());
        

        CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
            weight.device_type() == base::DeviceType::kDeviceCUDA &&
            output.device_type() == base::DeviceType::kDeviceCUDA);
        #if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
            const float eps = 1e-6f;
        #else
            const float eps = 1e-5f;
        #endif
        const int32_t size = static_cast<int32_t>(input.size());
        float* in_ptr = const_cast<float*>(input.ptr<float>());
        float* wei_ptr = const_cast<float*>(weight.ptr<float>());
        float* out_ptr = const_cast<float*>(output.ptr<float>());
        constexpr int threads_num = 128;
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            row_rmsnorm_f32<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
        } else {
            row_rmsnorm_f32<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
        }
    }

    void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream)
    {
        CHECK(!input.is_empty());
        CHECK(!weight.is_empty());
        CHECK(!output.is_empty());

        CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
            weight.device_type() == base::DeviceType::kDeviceCUDA &&
            output.device_type() == base::DeviceType::kDeviceCUDA); 

        const float eps = 1e-6f;

        //如果是2维，total_size是2；如果是3维，total_size是3；
        const int32_t total_size = static_cast<int32_t>(intput.size());
        //size 最后一维的大小
        const int32_t size = input.get_dim(input.dims_size() - 1);
        //向下取整，dim_size行数据需要归一化
        const int32_t dim_size = total_size / size;

        float* in_ptr = const_cast<float*>(input.ptr<float>());
        float* wei_ptr = const_cast<float*>(weight.ptr<float>());
        float* out_ptr = const_cast<float*>(output.ptr<float>());
        constexpr int threads_num = 128;
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size,
                                                                    size, eps);
        } else {
            row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
        }

    }


}