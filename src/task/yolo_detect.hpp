#pragma once

#include <vector>
#include <memory>

#include "types.hpp"
#include "engine.hpp"


struct Detection
{
    int id {-1};
    float score {0.f};
    Rect2f box;

    Detection() = default;
    Detection(int id, float score, const Rect2f& box) :
    id(id), score(score), box(box) {}
};


class YoloDetect : public Engine
{
public:
    using Result = std::vector<Detection>;
    using ResultPtr = std::unique_ptr<Result>;

    explicit YoloDetect(const std::string &modelPath, float scoreThres = 0.25f, float nmsThres = 0.7f);

    ResultPtr Predict(const void* data, size_t len);

    ResultPtr Postprocess(
        const rknn_tensor_mem* const* output,
        const rknn_tensor_attr* attr,
        const rknn_tensor_attr* nativeAttr,
        size_t num
    );

private:
    float _scoreThres;
    float _nmsThres;

    template<typename T>
    void _DecodeBunch(const rknn_tensor_mem* const* output,
                      const rknn_tensor_attr* attr,
                      const rknn_tensor_attr* nativeAttr,
                      std::vector<Rect2f> &boxes,
                      std::vector<float> &scores,
                      std::vector<int> &classes);
};
