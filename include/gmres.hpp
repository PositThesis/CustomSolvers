#pragma once
#include "arnoldi.hpp"
#include "iteration_result.hpp"
#include "qr.hpp"
#include "solve_least_squares.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <vector>

using Eigen::MatrixX;
using Eigen::VectorX;

template <typename Scalar>
IterationResult<Scalar> run_gmres_no_restart(MatrixX<Scalar> A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters) {
    assert(A.rows() == rhs.rows());

    VectorX<Scalar> initial_residual = rhs - A * x_0;
    Scalar rho_0 = initial_residual.norm();
    MatrixX<Scalar> V = initial_residual / initial_residual.norm();
    MatrixX<Scalar> H;

    IterationResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(1);

    for (Eigen::Index iter; iter < iters; iter++) {
        ArnoldiResult<Scalar> iter_result = arnoldi_step<Scalar>(V, A, rhs);
        H.conservativeResize(iter_resut.h_n.rows(), iter + 1);
        H.col(iter) = iter_result.h_n;
        V.conservativeResize(Eigen::NoChange, iter + 2);
        V.col(iter+1) = iter_result.v_next;

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    }

    for (Eigen::Index iter; iter < iters-1; iter++) {
        MatrixX<Scalar> H_sub = H.block(0, 0, iter+2, iter+1);
        VectorX<Scalar> z = solve_least_squares(H_sub, rho_0);
        VectorX<Scalar> x = x_0 + V.block(0, 0, V.rows(), iter+1) * z;
        Scalar residual = (rhs - x).norm();
        result.residuals.push_back(residual/rho_0);
    }

    return result;
}

template <typename Scalar>
IterationResult<Scalar> run_gmres_restart(MatrixX<Scalar> A, VectorX<Scalar> rhs, MatrixX<Scalar> x_0, Eigen::Index iters, Eigen::Index restart) {
    assert(A.rows() == rhs.rows());
    assert(restart > 0);

    VectorX<Scalar> initial_residual = rhs - A * x_0;

    MatrixX<Scalar> V = initial_residual / initial_residual.norm();
    MatrixX<Scalar> H;
    std::vector<MatrixX<Scalar>> Vs;
    std::vector<MatrixX<Scalar>> Hs;
    std::vector<Scalar> rho_0s;
    std::vector<VectorX<Scalar>> x_0s;


    IterationResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(1);

    for (Eigen::Index iter; iter < iters; iter++) {
        ArnoldiResult<Scalar> iter_result = arnoldi_step<Scalar>(V, A, rhs);
        H.conservativeResize(iter_resut.h_n.rows(), iter + 1);
        H.col(iter) = iter_result.h_n;
        V.conservativeResize(Eigen::NoChange, iter + 2);
        V.col(iter+1) = iter_result.v_next;

        // save data on restart and at the end
        if (iter % restart == 0 || iter == iters - 1) {
            // perform reset: store current data, compute new x_0, re-initialize variables
            MatrixX<Scalar> V_ = V;
            MatrixX<Scalar> H_ = H;

            Vs.push_back(V_);
            Hs.push_back(H_);
            rho_0s.push_back(initial_residual.norm());
            x_0s.push_back(x_0);

            VectorX<Scalar> z = solve_least_squares(H, rho_0);
            VectorX<Scalar> x = x_0 + V.block(0, 0, V.rows(), iter+1) * z;

            x_0 = x;
            initial_residual = rhs - A * x_0;
            H.resize(0, 0);
            V = initial_residual / initial_residual.norm();
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    }

    for (Eigen::Index iter; iter < iters-1; iter++) {
        MatrixX<Scalar> H_sub = H.block(0, 0, iter+2, iter+1);
        VectorX<Scalar> z = solve_least_squares(H_sub, rho_0);
        VectorX<Scalar> x = V.block(0, 0, V.rows(), iter+1) * z;
        Scalar residual = (rhs - x).norm();
        result.residuals.push_back(residual/rho_0);
    }

    return result;
}