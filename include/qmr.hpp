#pragma once
#include "lanczos_la.hpp"
#include "iteration_result.hpp"
#include "solve_least_squares.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <vector>

using Eigen::MatrixX;
using Eigen::VectorX;

template <typename Scalar, typename MatrixType>
SolverResult<Scalar> run_qmr_la(MatrixType A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters) {
    assert(A.rows() == rhs.rows());
    std::cout << "start QMR LA" << std::endl;

    VectorX<Scalar> initial_residual = rhs - A * x_0;
    Scalar rho_0 = initial_residual.norm();
    MatrixX<Scalar> V = initial_residual / initial_residual.norm();
    MatrixX<Scalar> W = V;
    MatrixX<Scalar> H;

    MatrixX<Scalar> V_k = initial_residual / initial_residual.norm();
    MatrixX<Scalar> W_k = V_k;

    MatrixX<Scalar> V_k_last, W_k_last, D_last;

    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(rho_0);

    Eigen::Index k = 0;

    for (Eigen::Index iter = 0; iter < iters; iter++) {
        LanczosLAIterResult<Scalar> iter_result;
        try {
            iter_result = lanczos_la_step<Scalar>(iter, k, V_k, W_k, V_k_last, W_k_last, D_last, A, rhs);
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

        if (iter % 10 == 0) {
            std::cout << "QMR LA iteration " << iter << " done" << std::endl;
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    }
    std::cout << "QMR LA iteration done" << std::endl;

    std::vector<Scalar> residuals = solve_all_least_squares<Scalar, MatrixType>(H, V, A, rhs, x_0, rho_0);
    for (uint64_t idx = 0; idx < residuals.size(); idx++) {
        result.residuals.push_back(residuals[idx]);
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}

