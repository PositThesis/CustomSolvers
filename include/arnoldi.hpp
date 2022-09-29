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

template <typename Scalar, typename MatrixType>
ArnoldiResult<Scalar> arnoldi_step(Eigen::Ref<MatrixX<Scalar>> V, Eigen::Ref<MatrixType> A, Eigen::Ref<VectorX<Scalar>> rhs) {
  VectorX<Scalar> v_next = A * V.col(V.cols() - 1);
  // removing the contribution in the direction of one vector might change the contributions in the directions of other vectors, cannot do all at once
  // VectorX<Scalar> h_n = V.adjoint() * v_next;
  VectorX<Scalar> h_n = VectorX<Scalar>::Zero(V.cols() + 1);
  for (Eigen::Index i = 0; i < V.cols(); i++) {
    h_n(i) = V.col(i).adjoint() * v_next;
    v_next -= h_n(i) * V.col(i);
  }
  h_n(V.cols()) = v_next.norm();
  v_next /= v_next.norm();

  ArnoldiResult<Scalar> result;
  result.v_next = v_next;
  result.h_n = h_n;
  return result;
}
