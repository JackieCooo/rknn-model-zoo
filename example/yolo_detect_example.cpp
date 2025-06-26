#include <string>
#include <cstring>
#include <cstdio>
#include <unistd.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "yolo_detect.hpp"
#include "label.hpp"
#include "rga.hpp"

#ifdef WITH_PREVIEW
    #include <sstream>
    #include <iomanip>

    #include "drawing.hpp"
    #include "display.hpp"
#endif


std::string modelPath;
std::string imagePath;
Label label;
float scoreThres = 0.25f;
float nmsThres = 0.7f;


int main(int argc, char* argv[])
{
    /* 解析命令行参数 */
    if (argc < 3) {
        std::printf("Usage: %s <model> <image> [-l label] [-s scoreThres] [-n nmsThres]\r\n", argv[0]);
        return -1;
    }

    modelPath.assign(argv[1]);
    imagePath.assign(argv[2]);

    int opt = -1;
    while ((opt = getopt(argc, argv, "l:s:n:")) != -1) {
        switch (static_cast<char>(opt))
        {
            /* 类别标签 */
            case 'l':
                label.Load(optarg);
                std::printf("loaded %ld labels\r\n", label.size());
                break;

            /* 分数阈值 */
            case 's':
                scoreThres = static_cast<float>(std::atof(optarg));
                break;

            /* NMS阈值 */
            case 'n':
                nmsThres = static_cast<float>(std::atof(optarg));
                break;

            default:
                break;
        }
    }

#ifdef WITH_PREVIEW
    /* 初始化屏幕 */
    Mpi::Init();
    Display display(VO_INTF_MIPI, {800, 1280, RK_FMT_RGBA8888, ROTATION_90}, 1, 2, 4);
    auto screen = display.GetFrame(0);
#endif

    /* 加载模型 */
    YoloDetect model(modelPath, scoreThres, nmsThres);
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
    std::printf("\r\n----- Got %ld objects -----\r\n", results->size());
    for (auto &&result : *results) {
#ifdef WITH_PREVIEW
        std::ostringstream oss;
        oss << label[result.id] << " @ " << std::setprecision(2) << std::fixed << result.score;
        Transformation trans({img.cols, img.rows}, inputSize);
        Rect box = trans.ToOriginal<float, int>(result.box);
        DrawBox(
            img,
            {
                box.x,
                box.y,
                box.width,
                box.height,
            },
            oss.str()
        );
#endif
        std::printf("%s [%.2f, %.2f, %.2f, %.2f] @ %.2f\r\n",
                    label[result.id].c_str(),
                    result.box.x,
                    result.box.y,
                    result.box.width,
                    result.box.height,
                    result.score);
    }

    std::printf("preprocess: %ld us, inference: %ld us, postprocess: %ld us\r\n",
                model.GetTimeCost().preprocess,
                model.GetTimeCost().inference,
                model.GetTimeCost().postprocess);

#ifdef WITH_PREVIEW
    rga->Run(
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
            screen->pData,
            Rga::Virtual,
            {
                1280,
                800,
                RK_FORMAT_RGBA_8888
            }
        }    
    );

    while (1);    
#endif

    return 0;
}
