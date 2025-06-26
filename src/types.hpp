#pragma once

#include <cstdint>
#include <memory>
#include <vector>


struct Size
{
    int width {0};
    int height {0};

    Size() = default;
    Size(int width, int height) :
    width(width), height(height) {}
    Size(int size) {width = size; height = size;}

    int size() const
    {
        return width * height;
    }
};

template<typename T>
struct Rect_
{
    T x;
    T y;
    T width;
    T height;

    Rect_() = default;
    Rect_(T x, T y, T width, T height) :
    x(x), y(y), width(width), height(height) {}
};

using Rect2i = Rect_<int>;
using Rect2f = Rect_<float>;
using Rect2d = Rect_<double>;
using Rect = Rect2i;

using Vec2f = std::vector<std::vector<float>>;
using Vec2i = std::vector<std::vector<int>>;

struct Transformation
{
    float scale {1.f};
    int xOff {0};
    int yOff {0};

    Transformation() = default;
    Transformation(float scale, int xOff, int yOff) :
    scale(scale), xOff(xOff), yOff(yOff) {}
    Transformation(const Size& src, const Size& dst)
    {
        scale = std::min(dst.width / 1.f / src.width, dst.height / 1.f / src.height);
        xOff = (dst.width - src.width * scale) / 2.f;
        yOff = (dst.height - src.height * scale) / 2.f;
    }

    template<typename T>
    void ToOriginal(Rect_<T>& rect)
    {
        rect.x = static_cast<T>((rect.x - xOff) / scale);
        rect.y = static_cast<T>((rect.y - yOff) / scale);
        rect.width = static_cast<T>(rect.width / scale);
        rect.height = static_cast<T>(rect.height / scale);
    }

    template<typename T, typename R>
    Rect_<R> ToOriginal(const Rect_<T>& rect)
    {
        return {
            static_cast<R>((rect.x - xOff) / scale),
            static_cast<R>((rect.y - yOff) / scale),
            static_cast<R>(rect.width / scale),
            static_cast<R>(rect.height / scale)
        };
    }

    template<typename T>
    void ToTarget(Rect_<T>& rect)
    {
        rect.x = static_cast<T>(rect.x * scale + xOff);
        rect.y = static_cast<T>(rect.y * scale + yOff);
        rect.width = static_cast<T>(rect.width * scale);
        rect.height = static_cast<T>(rect.height * scale);
    }

    template<typename T, typename R>
    Rect_<R> ToTarget(const Rect_<T>& rect)
    {
        return {
            static_cast<R>(rect.x * scale + xOff),
            static_cast<R>(rect.y * scale + yOff),
            static_cast<R>(rect.width * scale),
            static_cast<R>(rect.height * scale)
        };
    }
};

namespace Rknn
{
    struct Quantization
    {
        float scale {1.f};
        int zp {0};

        Quantization() = default;
        Quantization(float scale, int zp) : scale(scale), zp(zp) {}

        template<typename T>
        inline float Dequantize(T val)
        {
            return (static_cast<float>(val) - zp) * scale;
        }

        template<typename T>
        inline T Quantize(float val)
        {
            return static_cast<T>((val / scale) + zp);
        }

        template<typename T>
        static inline float Dequantize(T val, float scale, int32_t zp)
        {
            return (static_cast<float>(val) - zp) * scale;
        }

        template<typename T>
        static inline T Quantize(float val, float scale, int32_t zp)
        {
            return static_cast<T>((val / scale) + zp);
        }
    };
};
