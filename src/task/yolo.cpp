#include "yolo.hpp"
#include "utils/misc.hpp"

#include <cstdio>


Yolo::Yolo(const Size& size, float scoreThres, float nmsThres) : Task()
{
    SetParam("scoreThres", scoreThres);
    SetParam("nmsThres", nmsThres);
    SetParam("inputSize", size);
}

Yolo::ResultPtr Yolo::Postprocess(
    const rknn_tensor_mem* const* output,
    const rknn_tensor_attr* attr,
    const rknn_tensor_attr* nativeAttr,
    size_t num
)
{
    /* 输出包含6个张量，一共3组，每组2个，每组包含1个box和1个score输出 */
    /* (1, 64, 80, 80) (1, 80, 80, 80) (1, 64, 40, 40) (1, 80, 40, 40) (1, 64, 20, 20) (1, 80, 20, 20) */

    std::vector<Rect2f> boxes;  // 检测框
    std::vector<float> scores;  // 得分
    std::vector<int> classes;  // 类别
    auto type = attr[0].type;  // 数据类型
    uint32_t bunch = num / 2;  // 组数

    /* 遍历所有尺度输出 */
    for (uint32_t i = 0; i < bunch; i++) {
        if (type == RKNN_TENSOR_INT8) {
            _DecodeBunch<int8_t>(&output[2*i], &attr[2*i], &nativeAttr[2*i], boxes, scores, classes);
        } else if (type == RKNN_TENSOR_UINT8) {
            _DecodeBunch<uint8_t>(&output[2*i], &attr[2*i], &nativeAttr[2*i], boxes, scores, classes);
        } else if (type == RKNN_TENSOR_FLOAT32) {
            _DecodeBunch<float>(&output[2*i], &attr[2*i], &nativeAttr[2*i], boxes, scores, classes);
        }
    }

    /* NMS */
    auto nmsResult = Utils::NMS(boxes, scores, GetParam<float>("nmsThres"));

    /* 输出结果 */
    ResultPtr result = std::make_unique<Result>();
    for (auto &i : nmsResult) {
        result->emplace_back(
            Detection(
                classes[i],
                boxes[i],
                scores[i]
            )
        );
    }

    return result;
}

template<typename T>
void Yolo::_DecodeBunch(
    const rknn_tensor_mem* const* output,
    const rknn_tensor_attr* attr,
    const rknn_tensor_attr* nativeAttr,
    std::vector<Rect2f> &boxes,
    std::vector<float> &scores,
    std::vector<int> &classes)
{
    auto boxTensorShape = attr[0].dims;  // box矩阵shape
    uint32_t gridH = boxTensorShape[2];
    uint32_t gridW = boxTensorShape[3];
    uint32_t total = gridH * gridW;  /* box总数 */
    uint32_t dflLen = boxTensorShape[1] / 4;  /* DFL长度 */
    float scale = GetParam<Size>("inputSize").width / 1.f / gridW;  // 缩放比例
    uint32_t cls = attr[1].dims[1];  /* 类别数 */
    const T* boxTensor = static_cast<const T*>(output[0]->virt_addr);  /* (1, 4*dflLen, h, w) */
    const T* scoreTensor = static_cast<const T*>(output[1]->virt_addr);  /* (1, classes, h, w) */
    Rknn::Quantization boxQuant {attr[0].scale, attr[0].zp};  /* box矩阵量化参数 */
    Rknn::Quantization scoreQuant {attr[1].scale, attr[1].zp};  /* 分数量化参数 */
    T scoreThreshold = Rknn::Quantization::Quantize<T>(
        GetParam<float>("scoreThres"),
        scoreQuant.scale,
        scoreQuant.zp
    );  /* 量化后的分数阈值 */

    /* NC1HWC2转NCHW */
    if (nativeAttr[0].fmt == RKNN_TENSOR_NC1HWC2) {
        uint32_t boxTensorSize = output[0]->size;
        T *convertedBoxTensor = new T[boxTensorSize];
        Utils::NC1HWC2ToNCHW(boxTensor, convertedBoxTensor, &nativeAttr[0], &attr[0]);
        boxTensor = convertedBoxTensor;
    }
    if (nativeAttr[1].fmt == RKNN_TENSOR_NC1HWC2) {
        uint32_t scoreTensorSize = output[1]->size;
        T *convertedScoreTensor = new T[scoreTensorSize];
        Utils::NC1HWC2ToNCHW(scoreTensor, convertedScoreTensor, &nativeAttr[1], &attr[1]);
        scoreTensor = convertedScoreTensor;
    }

    /* 遍历所有box */
    for (uint32_t i = 0; i < gridH; i++) {
        for (uint32_t j = 0; j < gridW; j++) {
            /* 寻找最高得分类别 */
            uint32_t maxIndex = 0;
            T maxScore = scoreTensor[i * gridW + j];
            for (uint32_t k = 0, off = i * gridW + j; k < cls; k++, off += total) {
                if (scoreTensor[off] > maxScore) {
                    maxIndex = k;
                    maxScore = scoreTensor[off];
                }
            }

            /* 过滤低分框 */
            if (maxScore > scoreThreshold) {
                /* 计算box坐标 */
                float box[4] = {0};
                float dfl[boxTensorShape[1]] = {0};
                for (uint32_t k = 0, off = i * gridW + j; k < boxTensorShape[1]; k++, off += total) {
                    dfl[k] = boxQuant.Dequantize(boxTensor[off]);
                }
                Utils::DFL(dfl, dflLen, box);

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5f) * scale;
                y1 = (-box[1] + i + 0.5f) * scale;
                x2 = (box[2] + j + 0.5f) * scale;
                y2 = (box[3] + i + 0.5f) * scale;
                w = x2 - x1;
                h = y2 - y1;

                boxes.emplace_back(x1, y1, w, h);
                scores.push_back(scoreQuant.Dequantize(maxScore));
                classes.push_back(maxIndex);

                // std::printf("%d @ %.2f [%.2f %.2f %.2f %.2f]\r\n", classes.back(), scores.back(), x1, y1, w, h);
            }
        }
    }

    /* 释放资源 */
    if (nativeAttr[0].fmt == RKNN_TENSOR_NC1HWC2) {
        delete[] boxTensor;
        boxTensor = nullptr;
    }
    if (nativeAttr[1].fmt == RKNN_TENSOR_NC1HWC2) {
        delete[] scoreTensor;
        scoreTensor = nullptr;
    }
}

