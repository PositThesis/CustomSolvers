#pragma once
#include "lanczos_la.hpp"
#include "iteration_result.hpp"
#include "solve_least_squares.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <cmath>

using Eigen::MatrixX;
using Eigen::VectorX;

template <typename Scalar, typename MatrixType>
SolverResult<Scalar> run_qmr_la(MatrixType A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters, bool precond, int timeout) {
    assert(A.rows() == rhs.rows());
    std::cout << "start QMR LA" << std::endl;

    VectorX<Scalar> initial_residual = rhs - A * x_0;
    Scalar rho_0 = std::sqrt((Scalar)(initial_residual.adjoint()*initial_residual));

    auto [A_precond, rhs_precond] = precondition<Scalar, MatrixType>(A, rhs, precond);

    VectorX<Scalar> precond_initial_residual = rhs_precond - A_precond * x_0;
    Scalar precond_rho_0 = std::sqrt((Scalar)(precond_initial_residual.adjoint()*precond_initial_residual));
    MatrixX<Scalar> V = precond_initial_residual / precond_rho_0;
    MatrixX<Scalar> W = V;
    MatrixX<Scalar> H;

    MatrixX<Scalar> V_k = precond_initial_residual / precond_rho_0;
    MatrixX<Scalar> W_k = V_k;

    MatrixX<Scalar> V_k_last, W_k_last, D_last;

    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(rho_0);

    Eigen::Index k = 0;

    Eigen::Index last_successful_iteration = 0;

    for (Eigen::Index iter = 0; iter < iters; iter++) {
        LanczosLAIterResult<Scalar> iter_result;
        try {
            iter_result = lanczos_la_step<Scalar>(iter, k, V_k, W_k, V_k_last, W_k_last, D_last, A_precond, rhs_precond);
        } catch(bool fail) {
            std::cout << "QMR breakdown" << std::endl;
            break;
        }

        // always add to V, W and H
        H.conservativeResize(iter_result.h_n.rows(), H.cols() + 1);

        H.row(iter_result.h_n.rows() - 1).setZero();
        H.col(H.cols() - 1) = iter_result.h_n;

        V.conservativeResize(Eigen::NoChange, V.cols() + 1);
        V.col(V.cols() - 1) = iter_result.v_next;
        W.conservativeResize(Eigen::NoChange, W.cols() + 1);
        W.col(W.cols() - 1) = iter_result.w_next;

        if (iter_result.regular) {
            // reset blocks
            k++;
            V_k_last = V_k;
            W_k_last = W_k;
            D_last = W_k.adjoint() * V_k;

            V_k = iter_result.v_next;
            W_k = iter_result.w_next;
        } else {
            // extend blocks
            V_k.conservativeResize(Eigen::NoChange, V_k.cols() + 1);
            V_k.col(V_k.cols()-1) = iter_result.v_next;
            W_k.conservativeResize(Eigen::NoChange, W_k.cols() + 1);
            W_k.col(W_k.cols()-1) = iter_result.w_next;
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
        if (iter % 10 == 0) {
            long ms = std::chrono::duration_cast<std::chrono::milliseconds>(result.timestamps[iter + 1] - result.timestamps[std::max(0, (int)iter - 9)]).count();
            std::cout << "QMR LA iteration " << iter << " done; current rate is " << ms / (iter + 1 - std::max(0, (int)iter - 9)) << " ms per iteration" << std::endl;
        }
        if (std::chrono::duration_cast<std::chrono::seconds>(result.timestamps[iter + 1] - result.timestamps[0]).count() > timeout) {
            break;
        }
    }
    std::cout << "QMR LA iteration done" << std::endl;

    std::vector<Scalar> residuals = solve_all_least_squares<Scalar, MatrixType>(H, V, A, rhs, x_0, precond_rho_0);
    for (uint64_t idx = 0; idx < residuals.size(); idx++) {
        result.residuals.push_back(residuals[idx]);
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}

