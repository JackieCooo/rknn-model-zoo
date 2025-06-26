#pragma once


void DrawBox(
    cv::Mat& img,
    const cv::Rect& rect,
    const cv::String& text = "",
    const cv::Scalar& boxColor = {255, 0, 0},
    const cv::Scalar& textColor = {255, 255, 255}
);
