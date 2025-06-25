#ifndef __YOLO_H__
#define __YOLO_H__

#include "task.hpp"
#include "types.hpp"


class Yolo : public Task
{
public:
    struct Detection
    {
        int id;
        Rect2f rect;
        float score;

        Detection() = default;
        Detection(int id, const Rect2f& rect, float score) :
        id(id), rect(rect), score(score) {}
    };

    Yolo(const Size& size = {640, 640}, float scoreThres = 0.25f, float nmsThres = 0.7f);

    virtual ResultPtr Postprocess(
        const rknn_tensor_mem* const* output,
        const rknn_tensor_attr* attr,
        const rknn_tensor_attr* nativeAttr,
        size_t num
    ) override;

private:
    template<typename T>
    void _DecodeBunch(
        const rknn_tensor_mem* const* output,  // 该尺度下输出矩阵
        const rknn_tensor_attr* attr,  // 该尺度下矩阵属性
        const rknn_tensor_attr* nativeAttr,  // 该尺度下矩阵原始属性
        std::vector<Rect2f> &boxes,  // 检测框
        std::vector<float> &scores,  // 分数
        std::vector<int> &classes  // 类别
    );
};

#endif
