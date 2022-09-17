#pragma once

#include <Eigen/Dense>
#include <utility>
#include <cmath>

using Eigen::MatrixX;

using Eigen::Matrix2;

template <typename Scalar>
Matrix2<Scalar> make_rotation(Scalar x, Scalar y) {
//   Scalar angle = std::atan2(xy(1), yy(0));
//   Scalar sine = std::sin(angle);
//   Scalar cosine = std::cos(angle);
  Scalar denominator = std::sqrt(x * x + y * y)
  Scalar c = x / denominator;
  Scalar s = -y / denominator;

  Matrix2<Scalar> rotation;
  rotation << c, -s, s, c;
  return rotation;
}

template <typename Scalar>
struct QRDecompositionResult {
    MatrixX<Scalar> Q;
    MatrixX<Scalar> R;
}

template <typename Scalar>
std::pair<MatrixX<Scalar>, MatrixX<Scalar>> qr_decompose_hessenberg(MatrixX<Scalar> &H) {
  MatrixX<Scalar> Q = MatrixX<Scalar>::Identity(H.rows(), H.rows());
  MattixX<Scalar> R = H;

  for (Eigen::Index col = 0; col < H.cols(); col++) {
    Matrix2<Scalar> rot = make_rotation(R(col, col), R(col+1, col));
    R.block(col, col, 2, R.cols() - col) = rot * R.block(col, col, 2, R.cols() - col);
    Q.block(col, 0,   2, Q.cols())       = rot * Q.block(col, 0,   2, Q.cols()); // We can probably optimize the access here, most of Q is 0
  }

  QRDecompositionResult<Scalar> result;
  result.Q = Q;
  result.R = R;
  return result;
}

