#pragma once

#include <vector>
#include <expected>
#include <string>
#include <type_traits>
#include <random>

template <typename T>
    requires std::is_floating_point_v<T>
std::expected<std::vector<T>, std::string> randn(size_t n, T mean = T(0), T stddev = T(1))
{
    if (stddev <= T(0))
    {
        return std::unexpected("Standard deviation must be positive.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(mean, stddev);

    std::vector<T> result(n);

    for (auto &x : result)
    {
        x = dis(gen);
    }
    return std::expected<std::vector<T>, std::string>(result);
}
