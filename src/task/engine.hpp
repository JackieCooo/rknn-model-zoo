#pragma once

#include <string>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "rknn_api.h"
#include "arm_fp16.h"

#ifdef WITH_NEON
#include "arm_neon.h"
#endif

#include "types.hpp"


class Engine
{
public:
    struct TimeCost
    {
        int64_t preprocess {-1};
        int64_t inference {-1};
        int64_t postprocess {-1};
    };

    explicit Engine(const std::string &modelPath);
    ~Engine();

    void Init(const std::string &path);
    void Deinit();
    void AssignInput(const void *data, size_t len);
    int Inference();
    Size GetInputSize() const;
    const TimeCost& GetTimeCost() const;

protected:
    rknn_context _ctx = 0;
    rknn_tensor_mem **_inputMem = nullptr;
    rknn_tensor_mem **_outputMem = nullptr;
    uint32_t _inputNum = 0;
    uint32_t _outputNum = 0;
    rknn_tensor_attr *_inputAttr = nullptr;
    rknn_tensor_attr *_outputAttr = nullptr;
    rknn_tensor_attr *_inputNativeAttr = nullptr;
    rknn_tensor_attr *_outputNativeAttr = nullptr;

    TimeCost _timeCost;

    template<typename T>
    inline T& Input(int index)
    {
        return static_cast<T*>(_inputMem[0]->virt_addr)[index];
    }

    template<typename T>
    inline const T& Output(int index, int offset)
    {
        return static_cast<const T*>(_outputMem[index]->virt_addr)[offset];
    }

private:
    void _DumpTensorInfo(const char* tag, const rknn_tensor_attr *attr, int num);
};
