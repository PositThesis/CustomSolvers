#pragma once

#include <Eigen/Dense>
#include <Eigen/QR>
#include <utility>
#include <cmath>

using Eigen::MatrixX;

using Eigen::Matrix2;

template <typename Scalar>
Matrix2<Scalar> make_rotation(Scalar x, Scalar y) {
  Scalar denominator = std::sqrt(x * x + y * y);
  Scalar c, s;
  if (denominator == 0) {
    c = 1;
    s = 0;
  } else {
    c = x / denominator;
    s = -y / denominator;
  }

  Matrix2<Scalar> rotation;
  rotation << c, -s, s, c;
  return rotation;
}

template <typename Scalar>
struct QRDecompositionResult {
    MatrixX<Scalar> Q;
    MatrixX<Scalar> R;
};

template <typename Scalar>
QRDecompositionResult<Scalar> qr_decompose_hessenberg(MatrixX<Scalar> &H) {
  assert(H.rows() == H.cols() + 1);
  MatrixX<Scalar> Q = MatrixX<Scalar>::Identity(H.rows(), H.rows());
  MatrixX<Scalar> R = H;

  for (Eigen::Index col = 0; col < H.cols(); col++) {
    Matrix2<Scalar> rot = make_rotation(R(col, col), R(col+1, col));
    R.block(col, col, 2, R.cols() - col) = rot * R.block(col, col, 2, R.cols() - col);
    Q.block(col, 0,   2, Q.cols())       = rot * Q.block(col, 0,   2, Q.cols()); // We can probably optimize the access here, most of Q is 0
    if (col % 10 == 0) {
      std::cout << "QR decomposition step " << col << " / " << H.cols() << " done" << std::endl;
    }
  }
  std::cout << "QR decomposition done" << std::endl;

  QRDecompositionResult<Scalar> result;
  result.Q = Q.transpose();
  result.R = R;
  return result;
}

