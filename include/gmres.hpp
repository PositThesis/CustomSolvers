#pragma once
#include "arnoldi.hpp"
#include "householder_orthogonalization.hpp"
#include "iteration_result.hpp"
#include "qr.hpp"
#include "solve_least_squares.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <vector>

using Eigen::VectorX;

template <typename Scalar, typename MatrixType>
SolverResult<Scalar> run_gmres_no_restart(MatrixType A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters) {
    assert(A.rows() == rhs.rows());

    VectorX<Scalar> initial_residual = rhs - A * x_0;
    Scalar rho_0 = initial_residual.norm();
    MatrixX<Scalar> V = initial_residual / initial_residual.norm();
    MatrixX<Scalar> H = MatrixX<Scalar>::Zero(iters, iters+1);

    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(rho_0);

    for (Eigen::Index iter = 0; iter < iters; iter++) {
        ArnoldiResult<Scalar> iter_result = arnoldi_step<Scalar, MatrixType>(V, A, rhs);
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
            std::cout << "GMRES iteration " << iter << " / " << iters << " done" << std::endl;
        }
    }

    std::vector<Scalar> residuals = solve_all_least_squares<Scalar, MatrixType>(H, V, A, rhs, x_0, rho_0);
    for (uint64_t idx = 0; idx < residuals.size(); idx++) {
        result.residuals.push_back(residuals[idx]);
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}

template <typename Scalar, typename MatrixType>
SolverResult<Scalar> run_gmres_restart(MatrixType A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters, Eigen::Index restart) {
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
        ArnoldiResult<Scalar> iter_result = arnoldi_step<Scalar, MatrixType>(V, A, rhs);
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
            std::cout << "GMRES iteration " << iter << " / " << iters << " done" << std::endl;
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    }

    for (uint64_t block = 0; block < Vs.size(); block ++) {
        MatrixX<Scalar> V_ = Vs[block];
        MatrixX<Scalar> H_ = Hs[block];
        Scalar rho_0_ = rho_0s[block];
        VectorX<Scalar> x_0_ = x_0s[block];
        std::vector<Scalar> residuals = solve_all_least_squares<Scalar, MatrixType>(H_, V_, A, rhs, x_0_, rho_0_);
        for (uint64_t idx = 0; idx < residuals.size(); idx++) {
            result.residuals.push_back(residuals[idx]);
        }
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}

template <typename Scalar, typename MatrixType>
SolverResult<Scalar> run_gmres_householder_no_restart(MatrixType A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters) {
    assert(A.rows() == rhs.rows());
    assert(iters > 0);
    
    VectorX<Scalar> z = rhs - A * x_0;
    Scalar rho_0 = z.norm();
    Scalar beta;
    std::vector<VectorX<Scalar>> W;
    MatrixX<Scalar> H;
    MatrixX<Scalar> V;

    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(rho_0);

    for (Eigen::Index iter = 0; iter < iters + 1; iter++) {
        HouseholderResult<Scalar> iter_result = householder_step<Scalar, MatrixType>(W, A, z);
        std::cout << "finished iter" << std::endl;
        if (iter > 0) {
            // conservativeResize leaves the values unititialized, so set the newest row and col to 0 before inserting h_n
            H.conservativeResize(iter+1, iter);
            H.row(H.rows() - 1).setZero();
            H.col(H.cols() - 1).setZero();
            Eigen::Index h_length = std::min(iter_result.h_n_last.size(), iter+1);
            H.col(iter-1).head(h_length) = iter_result.h_n_last.head(h_length);
        }

        std::cout << "enlarged H" << std::endl;
        
        if (iter == 0) {
            beta = iter_result.beta;
        }

        if (iter < iters) {
            // the new column will be overridden immediately, no need to setZero
            V.conservativeResize(rhs.rows(), iter + 1);
            V.col(iter) = iter_result.v_n;

            W.push_back(iter_result.w_n);

            z = iter_result.z_n;
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
        if (iter % 10 == 0) {
            std::cout << "GMRES householder iteration " << iter << " / " << iters << " done" << std::endl;
        }
    }

    // there is a warning about beta being perhaps unititialized here. As long as the loop runs at least once, beta will be initialized
    // given the iters > 0 assertion above, this should always be the case.
    std::vector<Scalar> residuals = solve_all_least_squares<Scalar, MatrixType>(H, V, A, rhs, x_0, beta);
    for (uint64_t idx = 0; idx < residuals.size(); idx++) {
        result.residuals.push_back(residuals[idx]);
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}

template <typename Scalar, typename MatrixType>
SolverResult<Scalar> run_gmres_householder_restart(MatrixType A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters, Eigen::Index restart) {
    assert(A.rows() == rhs.rows());
    assert(restart > 0);

    VectorX<Scalar> z = rhs - A * x_0;

    std::vector<VectorX<Scalar>> W;
    MatrixX<Scalar> V;
    MatrixX<Scalar> H;
    std::vector<std::vector<VectorX<Scalar>>> Ws;
    std::vector<MatrixX<Scalar>> Vs;
    std::vector<MatrixX<Scalar>> Hs;
    std::vector<Scalar> betas;
    std::vector<VectorX<Scalar>> x_0s;
    Scalar beta;

    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(z.norm());

    Eigen::Index inner_iter = 0;
    for (Eigen::Index iter = 0; iter < iters+1; iter++) {
        HouseholderResult<Scalar> iter_result = householder_step<Scalar, MatrixType>(W, A, z);
        if (inner_iter > 0) {
            // conservativeResize leaves the values unititialized, so set the newest row and col to 0 before inserting h_n
            H.conservativeResize(inner_iter+1, inner_iter);
            H.row(H.rows() - 1).setZero();
            H.col(inner_iter-1).head(inner_iter+1) = iter_result.h_n_last.head(inner_iter+1);
        }

        if (inner_iter == 0) {
            beta = iter_result.beta;
            iter--; // don't count this as a step
        }

        if (inner_iter < iters) {
            // the new column will be overridden immediately, no need to setZero
            V.conservativeResize(rhs.rows(), inner_iter + 1);
            V.col(inner_iter) = iter_result.v_n;

            W.push_back(iter_result.w_n);

            z = iter_result.z_n;
        }


        inner_iter += 1;
        // save data on restart and at the end
        // restart increased by 1 because the first step in each round does not produce an H column
        if (inner_iter == restart+1 || iter == iters) {
            // perform reset: store current data, compute new x_0, re-initialize variables
            MatrixX<Scalar> V_ = V;
            MatrixX<Scalar> H_ = H;
            VectorX<Scalar> x_0_ = x_0;

            Vs.push_back(V_);
            Hs.push_back(H_);
            Ws.push_back(W);
            x_0s.push_back(x_0_);
            betas.push_back(beta);

            QRDecompositionResult<Scalar> qr = qr_decompose_hessenberg(H);
            // do not confuse this with the iteration z
            VectorX<Scalar> z_ = solve_least_squares<Scalar>(qr, beta, H.cols());
            VectorX<Scalar> x = x_0 + V.block(0, 0, V.rows(), inner_iter-1) * z_; // inner_iter, because we run inner_iter until restart+1

            x_0 = x;
            z = rhs - A * x_0;
            H.resize(0, 0);
            V.resize(0, 0);
            W = std::vector<VectorX<Scalar>>();

            inner_iter = 0;

        }

        if (iter % 10 == 0) {
            std::cout << "GMRES housholder iteration " << iter << " / " << iters << " done" << std::endl;
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    }

    for (uint64_t block = 0; block < Vs.size(); block ++) {
        MatrixX<Scalar> V_ = Vs[block];
        MatrixX<Scalar> H_ = Hs[block];
        Scalar beta_ = betas[block];
        VectorX<Scalar> x_0_ = x_0s[block];
        std::vector<Scalar> residuals = solve_all_least_squares<Scalar, MatrixType>(H_, V_, A, rhs, x_0_, beta_);
        for (uint64_t idx = 0; idx < residuals.size(); idx++) {
            result.residuals.push_back(residuals[idx]);
        }
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}

template <typename Scalar, typename MatrixType>
SolverResult<Scalar> run_gmres_householder_like_eigen_no_restart(MatrixType A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters) {
    assert(A.rows() == rhs.rows());

    VectorX<Scalar> r0 = rhs - A * x_0;

    std::vector<HouseholderData<Scalar>> hh_data;
    std::vector<VectorX<Scalar>> v_n;
    std::vector<VectorX<Scalar>> h_n;

    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(r0.norm());

    // make first hh vector
    hh_data.push_back(make_householder_vector_like_eigen<Scalar>(r0));

    // to signal whether we reached the end of the Krylov subspace
    // if we did not, we need to remove the last h_n from the residual computation
    bool complete_search = false;

    for(Eigen::Index iter = 0; iter < iters; iter++) {
        HouseholderStepResult<Scalar> hh_step;
        try {
            hh_step = householder_step_like_eigen<Scalar, MatrixType>(hh_data, A);
        } catch(bool fail) {
            break;
        }

        v_n.push_back(hh_step.v_n);
        if (!hh_step.last) {
            hh_data.push_back(hh_step.hh_data);
            h_n.push_back(hh_step.h_n);
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
        if (iter % 10 == 0) {
            std::cout << "GMRES eigen like householder iteration " << iter << " / " << iters << " done" << std::endl;
        }

        if (hh_step.last) {
            complete_search = true;
            break;
        }
    }
    std::cout << "GMRES eigen like householder finished" << std::endl;

    // assemble H and V matrices. Using two separate loops in case their size differs (due to some change)
    MatrixX<Scalar> V = MatrixX<Scalar>::Zero(A.rows(), v_n.size());
    for (uint64_t idx = 0; idx < v_n.size(); idx++) {
        V.col(idx) = v_n[idx];
    }

    if (complete_search) {
        h_n.pop_back();
    }
    MatrixX<Scalar> H = MatrixX<Scalar>::Zero(h_n.size() + 1, h_n.size());
    for (uint64_t idx = 0; idx < h_n.size(); idx++) {
        // skip the first one. That one is only for beta
        H.col(idx).head(idx+2) = h_n[idx].head(idx+2);
    }
    std::cout << "Created H: " << H.rows() << "x" << H.cols() << std::endl;

    std::vector<Scalar> residuals = solve_all_least_squares<Scalar, MatrixType>(H, V, A, rhs, x_0, hh_data[0].beta);
    for (uint64_t idx = 0; idx < residuals.size(); idx++) {
        result.residuals.push_back(residuals[idx]);
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}

template <typename Scalar, typename MatrixType>
SolverResult<Scalar> run_gmres_householder_like_eigen_restart(MatrixType A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Eigen::Index iters, Eigen::Index restart) {
    assert(A.rows() == rhs.rows());
    assert(restart > 0);

    VectorX<Scalar> r0 = rhs - A * x_0;

    std::vector<std::vector<HouseholderData<Scalar>>> hh_data;
    std::vector<std::vector<VectorX<Scalar>>> v_n;
    std::vector<std::vector<VectorX<Scalar>>> h_n;
    std::vector<VectorX<Scalar>> x_0s;

    // initialize first iteration
    hh_data.push_back(std::vector<HouseholderData<Scalar>>());
    v_n.push_back(std::vector<VectorX<Scalar>>());
    h_n.push_back(std::vector<VectorX<Scalar>>());
    x_0s.push_back(x_0);

    SolverResult<Scalar> result;
    result.timestamps.push_back(std::chrono::high_resolution_clock::now());
    result.residuals.push_back(r0.norm());

    // make first hh vector
    hh_data[0].push_back(make_householder_vector_like_eigen<Scalar>(r0));

    // to signal whether we reached the end of the Krylov subspace
    // if we did not, we need to remove the last h_n from the residual computation
    std::vector<bool> complete_search;
    complete_search.push_back(false);

    Eigen::Index inner_iter = 0;
    for(Eigen::Index iter = 0; iter < iters; iter++) {
        HouseholderStepResult<Scalar> hh_step;
        try {
            hh_step = householder_step_like_eigen<Scalar, MatrixType>(hh_data[hh_data.size()-1], A);
        } catch(bool fail) {
            break;
        }

        v_n[v_n.size() - 1].push_back(hh_step.v_n);
        if (!hh_step.last) {
            hh_data[hh_data.size() - 1].push_back(hh_step.hh_data);
            h_n[h_n.size() - 1].push_back(hh_step.h_n);
        }

        if (inner_iter == restart || hh_step.last) {
            // calculate intermediate result
            MatrixX<Scalar> V = MatrixX<Scalar>::Zero(A.rows(), v_n[v_n.size() - 1].size());
            for (uint64_t idx = 0; idx < v_n[v_n.size() - 1].size(); idx++) {
                V.col(idx) = v_n[v_n.size() - 1][idx];
            }

            int skip_last = 0;
            if (hh_step.last) {
                skip_last = 1;
            }
            MatrixX<Scalar> H = MatrixX<Scalar>::Zero(h_n[h_n.size() - 1].size() + 1 - skip_last, h_n[h_n.size() - 1].size() - skip_last);
            for (uint64_t idx = 0; idx < h_n[h_n.size() - 1].size() - skip_last; idx++) {
                // skip the first one. That one is only for beta
                H.col(idx).head(idx+2) = h_n[h_n.size() - 1][idx].head(idx+2);
            }

            QRDecompositionResult<Scalar> qr = qr_decompose_hessenberg(H);
            VectorX<Scalar> z = solve_least_squares<Scalar>(qr, hh_data[hh_data.size() - 1][0].beta, H.cols());
            VectorX<Scalar> x = x_0 + V.block(0, 0, V.rows(), z.size()) * z;

            // initialize next iteration
            hh_data.push_back(std::vector<HouseholderData<Scalar>>());
            v_n.push_back(std::vector<VectorX<Scalar>>());
            h_n.push_back(std::vector<VectorX<Scalar>>());
            x_0s.push_back(x);
            complete_search[complete_search.size() - 1] = hh_step.last;

            // set new guess
            x_0 = x;
            r0 = rhs - A * x_0;

            // make first hh vector
            hh_data[hh_data.size() - 1].push_back(make_householder_vector_like_eigen<Scalar>(r0));
            complete_search.push_back(false);

            inner_iter = 0;
        }

        result.timestamps.push_back(std::chrono::high_resolution_clock::now());
        if (iter % 10 == 0) {
            std::cout << "GMRES eigen like householder iteration " << iter << " / " << iters << " done" << std::endl;
        }
    }
    std::cout << "GMRES eigen like householder finished" << std::endl;

    for (uint64_t outer_iter = 0; outer_iter < hh_data.size(); outer_iter++) {
        MatrixX<Scalar> V = MatrixX<Scalar>::Zero(A.rows(), v_n[outer_iter].size());
        for (uint64_t idx = 0; idx < v_n[outer_iter].size(); idx++) {
            V.col(idx) = v_n[outer_iter][idx];
        }

        if (complete_search[outer_iter]) {
            h_n[outer_iter].pop_back();
        }
        MatrixX<Scalar> H = MatrixX<Scalar>::Zero(h_n[outer_iter].size() + 1, h_n[outer_iter].size());
        for (uint64_t idx = 0; idx < h_n[outer_iter].size(); idx++) {
            // skip the first one. That one is only for beta
            H.col(idx).head(idx+2) = h_n[outer_iter][idx].head(idx+2);
        }
        std::cout << "Created H: " << H.rows() << "x" << H.cols() << std::endl;

        std::vector<Scalar> residuals = solve_all_least_squares<Scalar, MatrixType>(H, V, A, rhs, x_0s[outer_iter], hh_data[outer_iter][0].beta);
        for (uint64_t idx = 0; idx < residuals.size(); idx++) {
            result.residuals.push_back(residuals[idx]);
        }
    }

    fill_durations<Scalar>(result);
    make_residuals_relative<Scalar>(result);

    return result;
}
