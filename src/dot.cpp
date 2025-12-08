#include "dot.h"

std::expected<float, std::string> dot_cuda(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size())
    {
        return std::unexpected<std::string>("Input vectors must have the same size.");
    }
    float result = dot_host(a, b);
    return std::expected<float, std::string>(result);
}