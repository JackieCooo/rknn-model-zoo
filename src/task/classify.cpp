#include "classify.hpp"

#include <algorithm>
#include <chrono>
#include <arm_fp16.h>


Classify::Classify(const std::string &modelPath, int topk) : Engine(modelPath), _topk(topk)
{

}

Classify::ResultPtr Classify::Predict(void* data, size_t len)
{
    /* 前处理 */
    auto t1 = std::chrono::high_resolution_clock::now();
    AssignInput(data, len);
    auto t2 = std::chrono::high_resolution_clock::now();
    _timeCost.preprocess = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    /* 执行推理 */
    auto t3 = std::chrono::high_resolution_clock::now();
    Inference();
    auto t4 = std::chrono::high_resolution_clock::now();
    _timeCost.inference = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    /* 后处理 */
    auto t5 = std::chrono::high_resolution_clock::now();
    auto result = Postprocess(_outputMem, _outputAttr, _outputNativeAttr, _outputNum);
    auto t6 = std::chrono::high_resolution_clock::now();
    _timeCost.postprocess = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

    return result;
}

Classify::ResultPtr Classify::Postprocess(
    const rknn_tensor_mem* const* output,
    const rknn_tensor_attr* attr,
    const rknn_tensor_attr* nativeAttr,
    size_t num
)
{
    /* (1, classNum) */
    uint32_t nc = attr[0].dims[1];
    auto type = attr[0].type;
    std::vector<Class> classes;

    /* 对结果排序 */
    for (uint32_t i = 0; i < nc; i++) {
        if (type == RKNN_TENSOR_FLOAT32) {
            classes.emplace_back(i, Output<float>(0, i));
        } else if (type == RKNN_TENSOR_INT8) {
            classes.emplace_back(i, Rknn::Quantization::Dequantize(Output<int8_t>(0, i), attr[0].scale, attr[0].zp));
        } else if (type == RKNN_TENSOR_FLOAT16) {
            classes.emplace_back(i, static_cast<float>(Output<float16_t>(0, i)));
        }
    }
    std::sort(classes.begin(),
              classes.end(),
              [](const Class& a, const Class& b) {
                    return a.score > b.score;
              });

    /* 取出topk结果 */
    _topk = _topk > static_cast<int>(nc) || _topk < 0 ? nc : _topk;
    classes.resize(_topk);

    return std::make_unique<Result>(classes);
}
