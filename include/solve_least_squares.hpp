#pragma once
#include "qr.hpp"
#include <Eigen/Dense>

using Eigen::VectorX;
using Eigen::MatrixX;

template <typename Scalar>
VectorX<Scalar> solve_least_squares(MatrixX<Scalar> &H, Scalar rho_0) {
    QRDecompositionResult<Scalar> qr = qr_decompose(H);
    MatrixX<Scalar> R = qr.R.block(0, 0, H.rows() - 1, H.cols());
    VectorX<Scalar> q = rho_0 * qr.Q.adjoint().col(0).block(0, H.rows()-1);

    VectorX<Scalar> z = VectorX<Scalar>::Zero(q.rows());

    // solve via forward substitution
    for (Eigen::Index k = z.rows() - 1; k >= 0; k--) {
        Scalar z_k = q(k);
        if (k < z.rows()) {
            z_k -= R.row(k).block(k+1, z.rows() - k).transpose() * z.block(k+1, z.rows() - k);
        }
        z_k /= R(k, k);
        z(k) = z_k;
    }
    return z;
}
