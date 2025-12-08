#pragma once

#include <vector>
#include <expected>
#include <string>

std::expected<float, std::string> dot_cuda(const std::vector<float> &a, const std::vector<float> &b);

float dot_host(const std::vector<float> &a, const std::vector<float> &b);