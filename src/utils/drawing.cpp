#include <opencv2/imgproc.hpp>

#include "drawing.hpp"


void DrawBox(
    cv::Mat& img,
    const cv::Rect& rect,
    const cv::String& text,
    const cv::Scalar& boxColor,
    const cv::Scalar& textColor
)
{
    /* 画框 */
    cv::rectangle(img, rect, boxColor);

    /* 文字 */
    constexpr int padding = 3;
    auto textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1., 2, nullptr);
    cv::rectangle(
        img,
        cv::Rect(
            rect.x,
            rect.y - textSize.height - padding * 2 < 0 ? rect.y : rect.y - textSize.height - padding * 2,
            textSize.width + padding * 2,
            textSize.height + padding * 2
        ),
        boxColor,
        cv::FILLED
    );
    cv::putText(
        img,
        text,
        {
            rect.x + padding,
            rect.y - textSize.height - padding * 2 < 0 ? rect.y + textSize.height + padding : rect.y - padding
        },
        cv::FONT_HERSHEY_SIMPLEX,
        1.,
        textColor,
        2
    );
}
