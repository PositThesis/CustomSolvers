#include <EigenIntegration/Overrides.hpp>
#include <EigenIntegration/MtxIO.hpp>
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include <universal/number/posit/posit.hpp>

#include <chrono>
#include <iostream>
#include <cstring>

#include "gmres.hpp"
// #include "qmr.hpp"
// #include "qmrwla.hpp"
#include "iteration_result.hpp"

using Eigen::MatrixX;
using Eigen::VectorX;

int main(int argc, char **argv) {
  std::string input_matrix("");
  std::string input_vector("");
  std::string output_file("");
  int iters = -1;
  int restart = -1;

  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "-im") == 0) {
      i += 1;
      input_matrix = std::string(argv[i]);
      continue;
    }
    if (std::strcmp(argv[i], "-iv") == 0) {
      i += 1;
      input_vector = std::string(argv[i]);
      continue;
    }
    if (std::strcmp(argv[i], "-o") == 0) {
      i += 1;
      output_file = std::string(argv[i]);
      continue;
    }
    if (std::strcmp(argv[i], "-iters") == 0) {
      i += 1;
      iters = std::stoi(argv[i]);
      continue;
    }
    if (std::strcmp(argv[i], "-restart") == 0) {
      i += 1;
      restart = std::stoi(argv[i]);
      continue;
    }
    std::cerr << "unknown option: " << argv[i] << std::endl;
    return -1;
  }

  assert(input_matrix.length() > 0);
  assert(input_vector.length() > 0);
  assert(output_file.length() > 0);
  (void)restart; // silence unused warnings

#ifdef USE_Float
  using Scalar = float;
#endif

#ifdef USE_Double
  using Scalar = double;
#endif

#ifdef USE_LongDouble
  using Scalar = long double;
#endif

#ifdef USE_Posit16
  using Scalar = sw::universal::posit<16, 2>;
#endif

#ifdef USE_Posit32
  using Scalar = sw::universal::posit<32, 2>;
#endif

#ifdef USE_Posit64
  using Scalar = sw::universal::posit<64, 2>;
#endif

  MatrixX<Scalar> A = get_matrix_from_mtx_file<Scalar>(input_matrix);
  VectorX<Scalar> rhs = get_matrix_from_mtx_file<Scalar>(input_vector);

#ifdef USE_GMRES
  SolverResult<Scalar> result;
  if (restart > 0) {
    result = run_gmres_restart<Scalar>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, restart);
  } else {
    result = run_gmres_householder_no_restart<Scalar>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters);
  }
#endif
#ifdef USE_QMR
  // SolverResult<Scalar> result = run_qmr_la<Scalar>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters);
#endif
#ifdef USE_QMRWLA
  // SolverResult<Scalar> result = run_qmr_wla<Scalar>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters);
#endif
  std::ofstream file(output_file + ".csv");
  file << "time [μs],iteration,residual" << std::endl;

  for (uint64_t i = 0; i < result.durations.size(); i++) {
    file << result.durations[i] << "," << i << "," << result.residuals[i] << std::endl;
    std::cout << "residual " << i << ": " << result.residuals[i] << std::endl;
  }

  file.close();
}
