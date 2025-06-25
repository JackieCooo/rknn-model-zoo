#ifndef __OPS_H__
#define __OPS_H__

#include <vector>

#include "rknn_api.h"

#include "types.hpp"
#include "rga.hpp"


namespace Utils
{
    void DFL(float* tensor, int len, float box[4]);
    float IoU(const Rect2f& b1, const Rect2f& b2);
    std::vector<int> NMS(const std::vector<Rect2f>& boxes, const std::vector<float>& scores, float threshold);
    void Transform(Rect2f& rect, Rga::Transformation& trans);

    template<typename T>
    void NC1HWC2ToNCHW(const T *src, T *dst, const rknn_tensor_attr* srcAttr, const rknn_tensor_attr* dstAttr)
    {
        uint32_t C2 = srcAttr->dims[4];
        uint32_t C = dstAttr->dims[1];
        uint32_t H = dstAttr->dims[2];
        uint32_t W = dstAttr->dims[3];
        uint32_t srcTotal = srcAttr->dims[2] * srcAttr->dims[3];
        uint32_t dstTotal = dstAttr->dims[2] * dstAttr->dims[3];

        for (uint32_t c = 0; c < C; ++c) {
            uint32_t plane  = c / C2;
            const T *scp = src + plane * srcTotal * C2;
            uint32_t off1 = c % C2;
            for (uint32_t h = 0; h < H; ++h) {
                for (uint32_t w = 0; w < W; ++w) {
                    uint32_t off2 = h * W + w;
                    dst[c * dstTotal + off2] = scp[C2 * off2 + off1];
                }
            }
        }
    }
};

#endif
