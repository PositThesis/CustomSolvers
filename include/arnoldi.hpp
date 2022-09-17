#pragma once

#include <Eigen/Dense>
#include <utility>

using Eigen::MatrixX;
using Eigen::VectorX;

template <typename Scalar>
struct ArnoldiResult {
    VectorX<Scalar> v_next;
    VectorX<Scalar> h_n;
};

template <typename Scalar>
ArnoldiResult<Scalar> arnoldi(MatrixX<Scalar> V, MatrixX<Scalar> A, VectorX<Scalar> rhs) {
  VectorX<Scalar> v_next = A * V.col(V.cols() - 1)
  // removing the contribution in the direction of one vector might change the contributions in the directions of other vectors, cannot do all at once
  // VectorX<Scalar> h_col = V.adjoint() * v_next;
  VectorX<Scalar> h_col = VectorX::Zero(V.cols() + 1);
  for (Eigen::Index i = 0; i < V.cols(); i++) {
    h_col(i) = V.col(i).adjoint() * v_next;
    v_next -= h_col(i) * V.col(i);
  }
  h_col(V.cols()) = v_next.norm();
  v_next /= v_next.norm();

  ArnoldiResult<Scalar> result;
  result.v_next = v_next;
  result.h_next = h_next;
  return result;
}
