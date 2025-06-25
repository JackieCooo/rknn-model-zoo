#pragma once

#include <vector>
#include <memory>

#include "engine.hpp"
#include "types.hpp"


struct Class
{
    uint32_t index {0};
    float score {0.f};

    Class() = default;
    Class(uint32_t index, float score) : index(index), score(score) {}
};


class Classify : public Engine
{
public:
    using Result = std::vector<Class>;
    using ResultPtr = std::unique_ptr<Result>;

    explicit Classify(const std::string &modelPath, int topk = 5);

    ResultPtr Predict(void* data, size_t len);

    ResultPtr Postprocess(
        const rknn_tensor_mem* const* output,
        const rknn_tensor_attr* attr,
        const rknn_tensor_attr* nativeAttr,
        size_t num
    );

private:
    int _topk;
};
