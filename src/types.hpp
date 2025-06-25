#ifndef __TYPES_H__
#define __TYPES_H__

#include <cstdint>
#include <memory>


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
struct Rect
{
    T x;
    T y;
    T width;
    T height;

    Rect() = default;
    Rect(T x, T y, T width, T height) :
    x(x), y(y), width(width), height(height) {}
};

using Rect2i = Rect<int>;
using Rect2f = Rect<float>;
using Rect2d = Rect<double>;

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

    struct Buffer
    {
        const void* addr;
        size_t len;

        Buffer() = default;
        Buffer(const void* addr, size_t len) : addr(addr), len(len) {}
    };

    using BufferPtr = std::unique_ptr<Buffer>;
};

#endif
