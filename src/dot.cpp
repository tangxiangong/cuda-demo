#include <thread>

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

std::expected<float, std::string> dot_cpu(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size())
    {
        return std::unexpected<std::string>("Input vectors must have the same size.");
    }

    size_t n = a.size();

    size_t num_threads = std::thread::hardware_concurrency();
    size_t chunk_size = (n + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<float> partial_sums(num_threads, 0.0f);

    for (size_t i = 0; i < num_threads; ++i)
    {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        threads.emplace_back([&, start, end, i]()
                             {
            float sum = 0.0f;
            for(size_t j = start; j < end; ++j) {
                sum += a[j] * b[j];
            }
            partial_sums[i] = sum; });
    }

    for (auto &thread : threads)
    {
        if (thread.joinable())
            thread.join();
    }

    float total_sum = 0.0f;
    for (const auto &partial_sum : partial_sums)
    {
        total_sum += partial_sum;
    }

    return std::expected<float, std::string>(total_sum);
}