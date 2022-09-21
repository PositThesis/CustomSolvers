#pragma once
#include "lanczos_wla.hpp"
#include "iteration_result.hpp"
#include "solve_least_squares.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <vector>

using Eigen::MatrixX;
using Eigen::VectorX;

template <typename Scalar>
SolverResult<Scalar> run_qmr_wla(MatrixX<Scalar> A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters) {
    assert(A.rows() == rhs.rows());
    std::cout << "start QMR WLA" << std::endl;

    VectorX<Scalar> initial_residual = rhs - A * x_0;
    Scalar rho_0 = initial_residual.norm();
    MatrixX<Scalar> V = initial_residual / initial_residual.norm();
    MatrixX<Scalar> W = V;
    MatrixX<Scalar> H;

    VectorX<Scalar> p_n, q_n;
    Scalar rho_n, xi_n, eps_n_last, beta_n_last;

    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(rho_0);

    for (Eigen::Index iter = 0; iter < iters; iter++) {
        LanczosWLAIterResult<Scalar> iter_result;
        try {
            iter_result = lanczos_wla_step<Scalar>(iter, V.col(V.cols() - 1), W.col(W.cols() - 1), p_n, q_n, xi_n, rho_n, eps_n_last, beta_n_last, A, rhs);
        } catch(bool fail) {
            std::cout << "QMR WLA breakdown" << std::endl;
            break;
        }

        // always add to V, W and H
        H.conservativeResize(iter_result.h_next.rows(), H.cols() + 1);

        H.row(iter_result.h_next.rows() - 1).setZero();
        H.col(H.cols() - 1) = iter_result.h_next;

        V.conservativeResize(Eigen::NoChange, V.cols() + 1);
        V.col(V.cols() - 1) = iter_result.v_next;

        W.conservativeResize(Eigen::NoChange, W.cols() + 1);
        W.col(W.cols() - 1) = iter_result.w_next;

        p_n = iter_result.p_n;
        q_n = iter_result.q_n;
        rho_n = iter_result.rho_next;
        xi_n = iter_result.xi_next;
        eps_n_last = iter_result.eps_n;
        beta_n_last = iter_result.beta_n;

        if (iter % 10 == 0) {
            std::cout << "QMR WLA iteration " << iter << " done" << std::endl;
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    }
    std::cout << "QMR WLA iteration done" << std::endl;

    std::vector<Scalar> residuals = solve_all_least_squares(H, V, A, rhs, x_0, rho_0);
    for (uint64_t idx = 0; idx < residuals.size(); idx++) {
        result.residuals.push_back(residuals[idx]);
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}

