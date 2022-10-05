#pragma once
#include <Eigen/IterativeLinearSolvers>
#include <utility>

using Eigen::VectorX;
using Eigen::DiagonalPreconditioner;

template <typename Scalar, typename MatrixType>
std::pair<MatrixType, VectorX<Scalar>> precondition(MatrixType A, VectorX<Scalar> rhs, bool precond) {
    if (!precond) {
        return std::pair(A, rhs);
    } else {
        // DiagonalPreconditioner<Scalar> pre(A);
        VectorX<Scalar> conditioner = A.diagonal();
        for (Eigen::Index idx = 0; idx < conditioner.size(); idx++) {
            if (conditioner.coeffRef(idx) != 0) {
                conditioner.coeffRef(idx) = 1 / conditioner.coeffRef(idx);
            } else {
                conditioner.coeffRef(idx) = Scalar(1);
            }
        }
        Eigen::SparseMatrix<Scalar> cond = Eigen::MatrixX<Scalar>(conditioner.asDiagonal()).sparseView();

        return std::pair(cond * A, cond * rhs);
    }
}