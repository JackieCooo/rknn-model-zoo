#include <string>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <unistd.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "classify.hpp"
#include "label.hpp"
#include "rga.hpp"


std::string modelPath;
std::string imagePath;
Label label;
int topk = -1;


int main(int argc, char* argv[])
{
    /* 解析命令行参数 */
    if (argc < 3) {
        std::printf("Usage: %s <model> <image> [-l label] [-k topk]\r\n", argv[0]);
        return -1;
    }

    modelPath.assign(argv[1]);
    imagePath.assign(argv[2]);

    int opt = -1;
    while ((opt = getopt(argc, argv, "l:")) != -1) {
        switch (static_cast<char>(opt))
        {
            /* 类别标签 */
            case 'l':
                label.Load(optarg);
                std::printf("loaded %ld labels\r\n", label.size());
                break;

            /* Topk */
            case 'k':
                topk = std::atoi(optarg);
                break;

            default:
                break;
        }
    }

    /* 加载模型 */
    Classify model(modelPath);
    auto inputSize = model.GetInputSize();

    /* 加载图片 */
    cv::Mat img = cv::imread(imagePath);
    std::printf("Read image %s\r\n", imagePath.c_str());
    auto& input = rga->Run(
        {
            (void*) img.data,
            Rga::Virtual,
            {
                img.cols,
                img.rows,
                RK_FORMAT_BGR_888
            }
        },
        {
            inputSize.width,
            inputSize.height,
            RK_FORMAT_RGB_888
        }
    );

    /* 获取结果 */
    auto results = model.Predict(input.addr, input.len);
    std::printf("\r\n----- Top %ld results -----\r\n", results->size());
    for (auto &&result : *results) {
        std::printf("%s @ %.2f\r\n", label[result.index].c_str(), result.score);
    }

    std::printf("preprocess: %ld us, inference: %ld us, postprocess: %ld us\r\n",
                model.GetTimeCost().preprocess,
                model.GetTimeCost().inference,
                model.GetTimeCost().postprocess);

    return 0;
}
