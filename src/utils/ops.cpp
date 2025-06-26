#include "ops.hpp"

#include <cmath>
#include <algorithm>
#include <set>


namespace Utils
{
    std::array<float, 4> DFL(const std::vector<float>& tensor)
    {
        size_t len = tensor.size() / 4;
        std::array<float, 4> box;

        for (int b = 0; b < 4; b++) {
            float exp_t[len];
            float exp_sum = 0;
            float acc_sum = 0;

            for (size_t i = 0; i < len; i++) {
                exp_t[i] = std::exp(tensor[i + b * len]);
                exp_sum += exp_t[i];
            }

            for (size_t i = 0; i < len; i++) {
                acc_sum += exp_t[i] / exp_sum * i;
            }

            box[b] = acc_sum;
        }

        return box;
    }

    float IoU(const Rect2f& b1, const Rect2f& b2)
    {
        float xmin1 = b1.x;
        float ymin1 = b1.y;
        float xmax1 = b1.x + b1.width;
        float ymax1 = b1.y + b1.height;

        float xmin2 = b2.x;
        float ymin2 = b2.y;
        float xmax2 = b2.x + b2.width;
        float ymax2 = b2.y + b2.height;

        float w = std::fmax(0.f, std::fmin(xmax1, xmax2) - std::fmax(xmin1, xmin2) + 1.0);
        float h = std::fmax(0.f, std::fmin(ymax1, ymax2) - std::fmax(ymin1, ymin2) + 1.0);
        float i = w * h;
        float u = (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) + (xmax2 - xmin2 + 1.0) * (ymax2 - ymin2 + 1.0) - i;

        return u <= 0.f ? 0.f : (i / u);    
    }

    std::vector<int> NMS(
        const std::vector<Rect2f>& boxes,
        const std::vector<float>& scores,
        const std::vector<int>& classes,
        float threshold
    )
    {
        std::vector<int> result;

        /* 统计类别数 */
        std::set<int> nc;
        for (size_t i = 0; i < classes.size(); i++) {
            nc.insert(classes[i]);
        }

        /* 每一类做NMS */
        for (auto& c : nc) {
            /* 选出该类下标 */
            std::vector<int> indice;
            for (size_t i = 0; i < classes.size(); i++) {
                if (classes[i] == c) {
                    indice.push_back(i);
                }
            }
            
            /* 按分数降序排序 */
            std::sort(indice.begin(), indice.end(), [&](const int& a, const int& b) {
                return scores[a] > scores[b];
            });

            /* 比对IoU，超过阈值的不要 */
            for (size_t i = 0; i < indice.size(); i++) {
                if (indice[i] == -1) {
                    continue;
                }

                for (size_t j = i + 1; j < indice.size(); j++) {
                    if (indice[j] == -1) {
                        continue;
                    }

                    float iou = IoU(boxes[indice[i]], boxes[indice[j]]);
                    if (iou > threshold) {
                        indice[j] = -1;
                    }
                }
            }

            /* 复制结果 */
            std::copy_if(indice.begin(), indice.end(), std::back_inserter(result), [](const int& v){return v != -1;});
        }

        return result;
    }
};
