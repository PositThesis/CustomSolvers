#pragma once
#include "arnoldi.hpp"
#include "householder_orthogonalization.hpp"
#include "iteration_result.hpp"
#include "qr.hpp"
#include "solve_least_squares.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <vector>

using Eigen::MatrixX;
using Eigen::VectorX;

template <typename Scalar>
SolverResult<Scalar> run_gmres_no_restart(MatrixX<Scalar> A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters) {
    assert(A.rows() == rhs.rows());

    VectorX<Scalar> initial_residual = rhs - A * x_0;
    Scalar rho_0 = initial_residual.norm();
    MatrixX<Scalar> V = initial_residual / initial_residual.norm();
    MatrixX<Scalar> H = MatrixX<Scalar>::Zero(iters, iters+1);

    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(rho_0);

    for (Eigen::Index iter = 0; iter < iters; iter++) {
        ArnoldiResult<Scalar> iter_result = arnoldi_step<Scalar>(V, A, rhs);
        // conservativeResize leaves the values unititialized, so set the newest row and col to 0 before inserting h_n
        H.conservativeResize(iter_result.h_n.rows(), iter + 1);
        H.row(iter_result.h_n.rows() - 1).setZero();
        H.col(iter).setZero();
        H.col(iter) = iter_result.h_n;

        // the new column will be overridden immediately, no need to setZero
        V.conservativeResize(Eigen::NoChange, iter + 2);
        V.col(iter+1) = iter_result.v_next;

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
        if (iter % 10 == 0) {
            std::cout << "GMRES iteration " << iter << " done" << std::endl;
        }
    }

    std::vector<Scalar> residuals = solve_all_least_squares(H, V, A, rhs, x_0, rho_0);
    for (uint64_t idx = 0; idx < residuals.size(); idx++) {
        result.residuals.push_back(residuals[idx]);
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}

template <typename Scalar>
SolverResult<Scalar> run_gmres_restart(MatrixX<Scalar> A, VectorX<Scalar> rhs, MatrixX<Scalar> x_0, Eigen::Index iters, Eigen::Index restart) {
    assert(A.rows() == rhs.rows());
    assert(restart > 0);

    VectorX<Scalar> initial_residual = rhs - A * x_0;

    MatrixX<Scalar> V = initial_residual / initial_residual.norm();
    MatrixX<Scalar> H;
    std::vector<MatrixX<Scalar>> Vs;
    std::vector<MatrixX<Scalar>> Hs;
    std::vector<Scalar> rho_0s;
    std::vector<VectorX<Scalar>> x_0s;


    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(initial_residual.norm());

    Eigen::Index inner_iter = 0;
    for (Eigen::Index iter = 0; iter < iters; iter++) {
        ArnoldiResult<Scalar> iter_result = arnoldi_step<Scalar>(V, A, rhs);
        H.conservativeResize(iter_result.h_n.rows(), (inner_iter + 1));
        H.row(iter_result.h_n.rows() - 1).setZero();
        H.col(inner_iter).setZero();
        H.col(inner_iter) = iter_result.h_n;
        V.conservativeResize(Eigen::NoChange, inner_iter + 2);
        V.col(inner_iter+1) = iter_result.v_next;

        inner_iter += 1;
        // save data on restart and at the end
        if (inner_iter == restart || iter == iters - 1) {
            // perform reset: store current data, compute new x_0, re-initialize variables
            MatrixX<Scalar> V_ = V;
            MatrixX<Scalar> H_ = H;
            VectorX<Scalar> x_0_ = x_0;

            Vs.push_back(V_);
            Hs.push_back(H_);
            rho_0s.push_back(initial_residual.norm());
            x_0s.push_back(x_0_);

            QRDecompositionResult<Scalar> qr = qr_decompose_hessenberg(H);
            VectorX<Scalar> z = solve_least_squares<Scalar>(qr, initial_residual.norm(), H.cols());
            VectorX<Scalar> x = x_0 + V.block(0, 0, V.rows(), inner_iter) * z; // inner_iter has already been increased this run

            x_0 = x;
            initial_residual = rhs - A * x_0;
            H.resize(0, 0);
            V = initial_residual / initial_residual.norm();

            inner_iter = 0;
        }

        if (iter % 10 == 0) {
            std::cout << "GMRES iteration " << iter << " done" << std::endl;
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    }

    for (uint64_t block = 0; block < Vs.size(); block ++) {
        MatrixX<Scalar> V_ = Vs[block];
        MatrixX<Scalar> H_ = Hs[block];
        Scalar rho_0_ = rho_0s[block];
        VectorX<Scalar> x_0_ = x_0s[block];
        std::vector<Scalar> residuals = solve_all_least_squares<Scalar>(H_, V_, A, rhs, x_0_, rho_0_);
        for (uint64_t idx = 0; idx < residuals.size(); idx++) {
            result.residuals.push_back(residuals[idx]);
        }
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}


template <typename Scalar>
SolverResult<Scalar> run_gmres_householder_no_restart(MatrixX<Scalar> A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters) {
    assert(A.rows() == rhs.rows());
    assert(iters > 0);
    
    VectorX<Scalar> z = rhs - A * x_0;
    Scalar rho_0 = z.norm();
    Scalar beta;
    MatrixX<Scalar> W;
    MatrixX<Scalar> H;
    MatrixX<Scalar> V;

    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(rho_0);

    for (Eigen::Index iter = 0; iter < iters + 1; iter++) {
        HouseholderResult<Scalar> iter_result = householder_step<Scalar>(W, A, z);
        if (iter > 0) {
            // conservativeResize leaves the values unititialized, so set the newest row and col to 0 before inserting h_n
            H.conservativeResize(std::max(rhs.rows(), iter+1), iter);
            if (H.rows() > rhs.rows()) {
                H.row(H.rows() - 1).setZero();
            }
            H.col(iter-1).head(rhs.rows()) = iter_result.h_n_last;
        }
        
        if (iter == 0) {
            beta = iter_result.beta;
        }

        if (iter < iters) {
            // the new column will be overridden immediately, no need to setZero
            V.conservativeResize(rhs.rows(), iter + 1);
            V.col(iter) = iter_result.v_n;

            W.conservativeResize(rhs.rows(), iter + 1);
            W.col(iter) = iter_result.w_n;

            z = iter_result.z_n;
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
        if (iter % 10 == 0) {
            std::cout << "GMRES householder iteration " << iter << " done" << std::endl;
        }
    }

    // there is a warning about beta being perhaps unititialized here. As long as the loop runs at least once, beta will be initialized
    // given the iters > 0 assertion above, this should always be the case.
    std::vector<Scalar> residuals = solve_all_least_squares(H, V, A, rhs, x_0, beta);
    for (uint64_t idx = 0; idx < residuals.size(); idx++) {
        result.residuals.push_back(residuals[idx]);
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}

template <typename Scalar>
SolverResult<Scalar> run_gmres_householder_restart(MatrixX<Scalar> A, VectorX<Scalar> rhs, MatrixX<Scalar> x_0, Eigen::Index iters, Eigen::Index restart) {
    assert(A.rows() == rhs.rows());
    assert(restart > 0);

    VectorX<Scalar> z = rhs - A * x_0;

    MatrixX<Scalar> W;
    MatrixX<Scalar> V;
    MatrixX<Scalar> H;
    std::vector<MatrixX<Scalar>> Ws;
    std::vector<MatrixX<Scalar>> Vs;
    std::vector<MatrixX<Scalar>> Hs;
    std::vector<Scalar> rho_0s;
    std::vector<Scalar> betas;
    std::vector<VectorX<Scalar>> x_0s;
    Scalar beta;


    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(z.norm());

    Eigen::Index inner_iter = 0;
    for (Eigen::Index iter = 0; iter < iters+1; iter++) {
        HouseholderResult<Scalar> iter_result = householder_step<Scalar>(W, A, z);
        if (inner_iter > 0) {
            // conservativeResize leaves the values unititialized, so set the newest row and col to 0 before inserting h_n
            H.conservativeResize(std::max(rhs.rows(), inner_iter+1), inner_iter);
            if (H.rows() > rhs.rows()) {
                H.row(H.rows() - 1).setZero();
            }
            H.col(inner_iter-1).head(rhs.rows()) = iter_result.h_n_last;
        }

        if (inner_iter == 0) {
            beta = iter_result.beta;
        }

        if (inner_iter < iters) {
            // the new column will be overridden immediately, no need to setZero
            V.conservativeResize(rhs.rows(), inner_iter + 1);
            V.col(inner_iter) = iter_result.v_n;

            W.conservativeResize(rhs.rows(), inner_iter + 1);
            W.col(inner_iter) = iter_result.w_n;

            z = iter_result.z_n;
        }


        inner_iter += 1;
        // save data on restart and at the end
        // restart increased by 1 because the first step in each round does not produce an H column
        if (inner_iter == restart+1 || iter == iters) {
            // perform reset: store current data, compute new x_0, re-initialize variables
            MatrixX<Scalar> V_ = V;
            MatrixX<Scalar> W_ = W;
            MatrixX<Scalar> H_ = H;
            VectorX<Scalar> x_0_ = x_0;

            Vs.push_back(V_);
            Hs.push_back(H_);
            rho_0s.push_back(z.norm());
            x_0s.push_back(x_0_);
            betas.push_back(beta);

            QRDecompositionResult<Scalar> qr = qr_decompose_hessenberg(H);
            // do not confuse this with the iteration z
            VectorX<Scalar> z_ = solve_least_squares<Scalar>(qr, beta, H.cols());
            VectorX<Scalar> x = x_0 + V.block(0, 0, V.rows(), inner_iter-1) * z_; // inner_iter we run inner_iter until restart+1

            x_0 = x;
            z = rhs - A * x_0;
            H.resize(0, 0);
            V.resize(0, 0);
            W.resize(0, 0);

            inner_iter = 0;

        }

        if (iter % 10 == 0) {
            std::cout << "GMRES housholder iteration " << iter << " done" << std::endl;
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    }

    for (uint64_t block = 0; block < Vs.size(); block ++) {
        MatrixX<Scalar> V_ = Vs[block];
        MatrixX<Scalar> H_ = Hs[block];
        Scalar beta_ = betas[block];
        VectorX<Scalar> x_0_ = x_0s[block];
        std::vector<Scalar> residuals = solve_all_least_squares<Scalar>(H_, V_, A, rhs, x_0_, beta_);
        for (uint64_t idx = 0; idx < residuals.size(); idx++) {
            result.residuals.push_back(residuals[idx]);
        }
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}
