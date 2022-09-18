#pragma once
#include <vector>
#include <chrono>

template <typename Scalar>
struct SolverResult {
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timestamps;
  std::vector<Scalar> residuals;
  std::vector<uint64_t> durations;
};

template <typename Scalar>
void fill_durations(SolverResult<Scalar> &sr) {
  sr.durations.clear();
  for (uint64_t i = 0; i < sr.timestamps.size(); i++) {
    sr.durations.push_back(std::chrono::duration_cast<std::chrono::microseconds>(sr.timestamps[i] - sr.timestamps[0]).count());
  }
}

template <typename Scalar>
void make_residuals_relative(SolverResult<Scalar> &sr) {
  Scalar rho_0 = sr.residuals[0];
  for (uint64_t idx = 0; idx < sr.residuals.size(); idx++) {
    sr.residuals[idx] /= rho_0;
  }
}