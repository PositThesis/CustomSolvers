#pragma once
#include "qr.hpp"
#include <Eigen/Dense>
#include <vector>

using Eigen::VectorX;
using Eigen::MatrixX;

// template <typename Scalar>
// VectorX<Scalar> solve_least_squares(MatrixX<Scalar> &H, Scalar rho_0) {
template <typename Scalar>
VectorX<Scalar> solve_least_squares(QRDecompositionResult<Scalar> &qr, Scalar rho_0, Eigen::Index size) {
    MatrixX<Scalar> R = qr.R.block(0, 0, size, size);
    VectorX<Scalar> q = rho_0 * qr.Q.transpose().block(0, 0, size, 1);

    VectorX<Scalar> z = VectorX<Scalar>::Zero(q.rows());

    // solve via forward substitution
    for (Eigen::Index k = z.rows() - 1; k >= 0; k--) {
        Scalar z_k = q(k);
        if (k < z.rows() - 1) {
            MatrixX<Scalar> m_ = R.block(k, k+1, 1, z.rows() - k - 1) * z.block(k+1, 0, z.rows() - k - 1, 1);
            assert(m_.rows() == 1 && m_.cols() == 1);
            z_k -= m_(0, 0);
        }
        // when the subspace is larger than the problem
        if (R(k, k) == 0) {
            z(k) = 0;
        } else {
            z_k /= R(k, k);
            z(k) = z_k;
        }
    }
    return z;
}

template <typename Scalar>
std::vector<Scalar> solve_all_least_squares(MatrixX<Scalar> &H, MatrixX<Scalar> &V, MatrixX<Scalar> &A, VectorX<Scalar> rhs, VectorX<Scalar> x_0, Scalar rho_0) {
    std::vector<Scalar> residuals;
    QRDecompositionResult<Scalar> qr = qr_decompose_hessenberg(H);

    for (Eigen::Index iter = 0; iter < H.cols(); iter++) {
        // MatrixX<Scalar> H_sub = H.block(0, 0, iter+2, iter+1);
        VectorX<Scalar> z = solve_least_squares<Scalar>(qr, rho_0, iter+1);
        VectorX<Scalar> x = x_0 + V.block(0, 0, V.rows(), iter+1) * z;

        Scalar residual = (rhs - A * x).norm();
        residuals.push_back(residual);
        if (iter % 10 == 0) {
            std::cout << "residual " << iter << " / " << H.cols() << " done" << std::endl;
        }
    }
    std::cout << "rediduals done" << std::endl;

    return residuals;
}