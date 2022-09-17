#pragma once
#include <vector>
#include <chrono>

template <typename Scalar>
struct SolverResult {
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timestamps;
  std::vector<Scalar> residuals;
}

