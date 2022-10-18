#pragma once
#include "lanczos_wla.hpp"
#include "iteration_result.hpp"
#include "solve_least_squares.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <vector>

using Eigen::MatrixX;
using Eigen::VectorX;

template <typename Scalar, typename MatrixType>
SolverResult<Scalar> run_qmr_wla(MatrixType A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters, bool precond, int timeout) {
    assert(A.rows() == rhs.rows());
    std::cout << "start QMR WLA" << std::endl;

    VectorX<Scalar> initial_residual = rhs - A * x_0;
    Scalar rho_0 = std::sqrt((Scalar)(initial_residual.adjoint()*initial_residual));

    auto [A_precond, rhs_precond] = precondition<Scalar, MatrixType>(A, rhs, precond);

    VectorX<Scalar> precond_initial_residual = rhs_precond - A_precond * x_0;
    Scalar precond_rho_0 = std::sqrt((Scalar)(precond_initial_residual.adjoint()*precond_initial_residual));

    // the runtime of this algorithm was dominated by the cost of moving the matrices, so preallocate
    MatrixX<Scalar> V = MatrixX<Scalar>::Zero(A.rows(), iters+1);
    V.col(0) = precond_initial_residual / precond_rho_0;
    MatrixX<Scalar> W = V;
    MatrixX<Scalar> H = MatrixX<Scalar>::Zero(iters + 1, iters);

    VectorX<Scalar> p_n, q_n;
    Scalar rho_n, xi_n, eps_n_last, beta_n_last;

    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(rho_0);

    Eigen::Index last_successful_iter = 0;

    for (Eigen::Index iter = 0; iter < iters; iter++) {
        LanczosWLAIterResult<Scalar> iter_result;
        try {
            iter_result = lanczos_wla_step<Scalar, MatrixType>(iter, V.col(iter), W.col(iter), p_n, q_n, xi_n, rho_n, eps_n_last, beta_n_last, A_precond, rhs_precond);
        } catch(bool fail) {
            std::cout << "QMR WLA breakdown" << std::endl;
            break;
        }

        H.col(iter).head(iter_result.h_next.size()) = iter_result.h_next;
        // these had one initial column
        V.col(iter+1) = iter_result.v_next;
        W.col(iter+1) = iter_result.w_next;

        p_n = iter_result.p_n;
        q_n = iter_result.q_n;
        rho_n = iter_result.rho_next;
        xi_n = iter_result.xi_next;
        eps_n_last = iter_result.eps_n;
        beta_n_last = iter_result.beta_n;

        last_successful_iter = iter;

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
        if (iter % 10 == 0) {
            int ms = std::chrono::duration_cast<std::chrono::milliseconds>(result.timestamps[iter + 1] - result.timestamps[std::max(0, (int)iter - 9)]).count();
            std::cout << "QMR WLA iteration " << iter << " done; current rate is " << ms / (iter + 1 - std::max(0, (int)iter - 9)) << " ms per iteration" << std::endl;
        }
        if (std::chrono::duration_cast<std::chrono::seconds>(result.timestamps[iter + 1] - result.timestamps[0]).count() > timeout) {
            break;
        }
    }
    std::cout << "QMR WLA iteration done" << std::endl;

    MatrixX<Scalar> V_ = V.block(0, 0, V.rows(), last_successful_iter + 1);
    MatrixX<Scalar> H_ = H.block(0, 0, last_successful_iter + 2, last_successful_iter + 1);

    std::cout << "made blocks" << std::endl;

    std::vector<Scalar> residuals = solve_all_least_squares<Scalar, MatrixType>(H_, V_, A, rhs, x_0, precond_rho_0);
    for (uint64_t idx = 0; idx < residuals.size(); idx++) {
        result.residuals.push_back(residuals[idx]);
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}

