#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <EigenIntegration/Overrides.hpp>
#include <EigenIntegration/std_integration.hpp>
#include <EigenIntegration/MtxIO.hpp>
#include <Eigen/Sparse>

#include <universal/number/posit/posit.hpp>

#include <chrono>
#include <iostream>
#include <cstring>

#include "gmres.hpp"
#include "qmr.hpp"
#include "qmr_wla.hpp"
#include "iteration_result.hpp"

using Eigen::MatrixX;
using Eigen::SparseMatrix;
using Eigen::VectorX;

int main(int argc, char **argv) {
  std::string input_matrix("");
  std::string input_vector("");
  std::string output_file("");
  int iters = -1;
  int restart = -1;
  bool gmres_householder = false;
  bool sparse = false;
  bool eigen_like = false;
  bool precond = false;

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
    if (std::strcmp(argv[i], "-hh") == 0) {
      gmres_householder = true;
      continue;
    }
    if (std::strcmp(argv[i], "-sparse") == 0) {
      sparse = true;
      continue;
    }
    if (std::strcmp(argv[i], "-eigen_like") == 0) {
      eigen_like = true;
      continue;
    }
    if (std::strcmp(argv[i], "-precond") == 0) {
      precond = true;
      continue;
    }
    std::cerr << "unknown option: " << argv[i] << std::endl;
    return -1;
  }

  assert(input_matrix.length() > 0);
  assert(input_vector.length() > 0);
  assert(output_file.length() > 0);
  (void)restart; // silence unused warnings
  (void)gmres_householder;
  (void)eigen_like;

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
#ifdef USE_PositCustom
  using Scalar = sw::universal::posit<USE_Posit_NBits, USE_Posit_ES>;
#endif

  MatrixX<Scalar> A = get_matrix_from_mtx_file<Scalar>(input_matrix);
  VectorX<Scalar> rhs = get_matrix_from_mtx_file<Scalar>(input_vector);

#ifdef USE_GMRES
  SolverResult<Scalar> result;
  if (eigen_like) {
    assert(gmres_householder);
    if (sparse) {
      if (restart > 0) {
        result = run_gmres_householder_like_eigen_restart<Scalar, SparseMatrix<Scalar>>(A.sparseView(), rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, restart, precond);
      } else {
        result = run_gmres_householder_like_eigen_no_restart<Scalar, SparseMatrix<Scalar>>(A.sparseView(), rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, precond);
      }
    } else {
      if (restart > 0) {
        result = run_gmres_householder_like_eigen_restart<Scalar, MatrixX<Scalar>>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, restart, precond);
      } else {
        result = run_gmres_householder_like_eigen_no_restart<Scalar, MatrixX<Scalar>>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, precond);
      }
    }
  } else {
    if (sparse) {
      if (restart > 0) {
        if (gmres_householder) {
          result = run_gmres_householder_restart<Scalar, SparseMatrix<Scalar>>(A.sparseView(), rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, restart, precond);
        } else {
          result = run_gmres_restart<Scalar, SparseMatrix<Scalar>>(A.sparseView(), rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, restart, precond);
        }
      } else {
        if (gmres_householder) {
          result = run_gmres_householder_no_restart<Scalar, SparseMatrix<Scalar>>(A.sparseView(), rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, precond);
        } else {
          result = run_gmres_no_restart<Scalar, SparseMatrix<Scalar>>(A.sparseView(), rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, precond);
        }
      }
    } else {
      if (restart > 0) {
        if (gmres_householder) {
          result = run_gmres_householder_restart<Scalar, MatrixX<Scalar>>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, restart, precond);
        } else {
          result = run_gmres_restart<Scalar, MatrixX<Scalar>>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, restart, precond);
        }
      } else {
        if (gmres_householder) {
          result = run_gmres_householder_no_restart<Scalar, MatrixX<Scalar>>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, precond);
        } else {
          result = run_gmres_no_restart<Scalar, MatrixX<Scalar>>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, precond);
        }
      }
    }
  }
#endif
#ifdef USE_QMR
  SolverResult<Scalar> result;
  if (sparse) {
    result = run_qmr_la<Scalar, SparseMatrix<Scalar>>(A.sparseView(), rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, precond);
  } else {
    result = run_qmr_la<Scalar, MatrixX<Scalar>>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, precond);
  }
#endif
#ifdef USE_QMRWLA
  SolverResult<Scalar> result;
  if (sparse) {
    result = run_qmr_wla<Scalar, SparseMatrix<Scalar>>(A.sparseView(), rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, precond);
  } else {
    result = run_qmr_wla<Scalar, MatrixX<Scalar>>(A, rhs, VectorX<Scalar>::Zero(rhs.rows()), iters, precond);
  }
#endif
  std::ofstream file(output_file + ".csv");
  file << "time [μs],iteration,residual" << std::endl;

  for (uint64_t i = 0; i < result.residuals.size(); i++) {
    file << result.durations[i] << "," << i << "," << result.residuals[i] << std::endl;
  }

  file.close();
}

