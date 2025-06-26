#include <filesystem>
#include <chrono>

#include "arm_fp16.h"

#ifdef WITH_NEON
    #include "arm_neon.h"
#endif

#include "engine.hpp"


Engine::Engine(const std::string &modelPath)
{
    Init(modelPath);
}

Engine::~Engine()
{
    Deinit();
}

void Engine::Init(const std::string &path)
{
    if (!std::filesystem::exists(path)) {
        std::printf("model %s not exist\r\n", path.c_str());
        return;
    }

    int ret = RKNN_SUCC;

    /* 初始化上下文 */
    ret = rknn_init(
        &_ctx,
        static_cast<void *>(const_cast<char *>(path.c_str())),
        0,
        RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU,
        nullptr
    );
    if (ret != RKNN_SUCC) {
        std::printf("RKNN init failed\r\n");
        return;
    }

    /* 配置多核 */
    ret = rknn_set_core_mask(_ctx, RKNN_NPU_CORE_ALL);
    if (ret != RKNN_SUCC) {
        std::printf("RKNN set core mask failed\r\n");
    }

    /* 获取输入输出张量数量 */
    rknn_input_output_num ioNum;
    ret = rknn_query(_ctx, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(rknn_input_output_num));
    if (ret != RKNN_SUCC) {
        std::printf("query input output num failed\r\n");
        return;
    }
    _inputNum = ioNum.n_input;
    _outputNum = ioNum.n_output;

    /* 分配输入张量内存 */
    _inputNativeAttr = new rknn_tensor_attr[_inputNum];
    _inputMem = new rknn_tensor_mem*[_inputNum];
    for (uint32_t i = 0; i < _inputNum; i++) {
        _inputNativeAttr[i].index = i;
        ret = rknn_query(_ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &_inputNativeAttr[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            std::printf("query input %d native attribute failed\r\n", i);
        }

        _inputMem[i] = rknn_create_mem(_ctx, _inputNativeAttr[i].size_with_stride);
        if (_inputMem == nullptr) {
            std::printf("allocate input %d memory failed\r\n", i);
        }

        ret = rknn_set_io_mem(_ctx, _inputMem[i], &_inputNativeAttr[i]);
        if (ret != RKNN_SUCC) {
            std::printf("set input %d io mem failed\r\n", i);
        }
    }
    _DumpTensorInfo("Input tensor native attribute", _inputNativeAttr, _inputNum);

    /* 分配输出张量内存 */
    _outputNativeAttr = new rknn_tensor_attr[_outputNum];
    _outputMem = new rknn_tensor_mem*[_outputNum];
    for (uint32_t i = 0; i < _outputNum; i++) {
        _outputNativeAttr[i].index = i;
        ret = rknn_query(_ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &_outputNativeAttr[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            std::printf("query output %d native attribute failed\r\n", i);
        }

        _outputMem[i] = rknn_create_mem(_ctx, _outputNativeAttr[i].size_with_stride);
        if (_outputMem == nullptr) {
            std::printf("allocate output %d memory failed\r\n", i);
        }

        ret = rknn_set_io_mem(_ctx, _outputMem[i], &_outputNativeAttr[i]);
        if (ret != RKNN_SUCC) {
            std::printf("set output %d io mem failed\r\n", i);
        }
    }
    _DumpTensorInfo("Output tensor native attribute", _outputNativeAttr, _outputNum);

    /* 获取张量信息 */
    _inputAttr = new rknn_tensor_attr[_inputNum];
    for (uint32_t i = 0; i < _inputNum; i++) {
        _inputAttr[i].index = i;
        ret = rknn_query(_ctx, RKNN_QUERY_INPUT_ATTR, &_inputAttr[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            std::printf("query input %d attribute failed\r\n", i);
        }
    }
    _DumpTensorInfo("Input tensor attribute", _inputAttr, _inputNum);

    _outputAttr = new rknn_tensor_attr[_outputNum];
    for (uint32_t i = 0; i < _outputNum; i++) {
        _outputAttr[i].index = i;
        ret = rknn_query(_ctx, RKNN_QUERY_OUTPUT_ATTR, &_outputAttr[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            std::printf("query output %d attribute failed\r\n", i);
        }
    }
    _DumpTensorInfo("Output tensor attribute", _outputAttr, _outputNum);
}

void Engine::Deinit()
{
    if (_inputMem) {
        for (uint32_t i = 0; i < _inputNum; i++) {
            rknn_destroy_mem(_ctx, _inputMem[i]);
            _inputMem[i] = nullptr;
        }
        delete[] _inputMem;
        _inputMem = nullptr;
    }
    if (_outputMem) {
        for (uint32_t i = 0; i < _outputNum; i++) {
            rknn_destroy_mem(_ctx, _outputMem[i]);
            _outputMem[i] = nullptr;
        }
        delete[] _outputMem;
        _outputMem = nullptr;
    }

    if (_inputAttr) {
        delete[] _inputAttr;
        _inputAttr = nullptr;
    }
    if (_outputAttr) {
        delete[] _outputAttr;
        _outputAttr = nullptr;
    }

    if (_inputNativeAttr) {
        delete[] _inputNativeAttr;
        _inputNativeAttr = nullptr;
    }
    if (_outputNativeAttr) {
        delete[] _outputNativeAttr;
        _outputNativeAttr = nullptr;
    }

    rknn_destroy(_ctx);
}

void Engine::AssignInput(const void *data, size_t len)
{
    const uint8_t *sp = (const uint8_t *) data;

#if (defined WITH_NEON && defined __ARM_NEON)
    /* NEON指令集加速 */
    uint8_t *dp = (uint8_t *) _inputMem[0]->virt_addr;
    uint8x16_t sub = vdupq_n_u8(128);  // 加载被减数
    for (size_t i = 0; i < len / 16; i++, sp += 16, dp += 16) {
        uint8x16_t u8x16 = vld1q_u8(sp);  // 加载16个uint8
        u8x16 = vsubq_u8(u8x16, sub);  // 作减法
        vst1q_u8(dp, u8x16);  // 导出结果
    }
#else
    for (size_t i = 0; i < len; i++) {
        Input<int8_t>(i) = (sp[i] - 128);
    }
#endif
}

int Engine::Inference()
{
    auto t1 = std::chrono::high_resolution_clock::now();
    int ret = rknn_run(_ctx, nullptr);
    auto t2 = std::chrono::high_resolution_clock::now();
    _timeCost.inference = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    if (ret != RKNN_SUCC) {
        std::printf("Failed to invoke RKNN\r\n");
    }
    return ret;
}

Size Engine::GetInputSize() const
{
    if (_inputAttr[0].fmt == RKNN_TENSOR_NCHW) {
        return {
            static_cast<int>(_inputAttr[0].dims[3]),
            static_cast<int>(_inputAttr[0].dims[2])
        };
    } else if (_inputAttr[0].fmt == RKNN_TENSOR_NHWC) {
        return {
            static_cast<int>(_inputAttr[0].dims[2]),
            static_cast<int>(_inputAttr[0].dims[1])
        };
    } else {
        return {};
    }
}

const Engine::TimeCost& Engine::GetTimeCost() const
{
    return _timeCost;
}

void Engine::_DumpTensorInfo(const char* tag, const rknn_tensor_attr *attr, int num)
{
    std::printf("%s:\r\n", tag);
    for (int i = 0; i < num; i++) {
        std::printf("name: %s, dim: [", attr[i].name);
        for (uint32_t j = 0; j < attr[i].n_dims; j++) {
            std::printf("%d", attr[i].dims[j]);
            if (j != attr[i].n_dims - 1) {
                std::printf(" ");
            }
        }
        std::printf("], dtype: %s, format: %s, size: %d, wstride: %d, hstride: %d, stride_size: %d, scale: %f, zp: %d\r\n",
                    get_type_string(attr[i].type),
                    get_format_string(attr[i].fmt),
                    attr[i].size,
                    attr[i].w_stride,
                    attr[i].h_stride,
                    attr[i].size_with_stride,
                    attr[i].scale,
                    attr[i].zp);
    }
}
