#pragma once
#include "lanczos_la.hpp"
#include "iteration_result.hpp"
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
    MatrixX<Scalar> W = V;
    MatrixX<Scalar> H;

    IterationResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(1);

    for (Eigen::Index iter; iter < iters; iter++) {
        LanczosLAResult<Scalar> iter_result = lanczos_la_step<Scalar>(V, A, rhs);
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

