//
// Created by fss on 23-6-4.
//
#include "include/tensor/tensor.h"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_tensor, create_cube) {
  using namespace kuiper_infer;
  int32_t size = 27;
  std::vector<float> datas;
  for (int i = 0; i < size; ++i) {
    datas.push_back(float(i));
  }
  arma::Cube<float> cube(3, 3, 3);
  memcpy(cube.memptr(), datas.data(), size * sizeof(float));
  LOG(INFO) << cube;
}

TEST(test_tensor, create_1dtensor) {
  using namespace kuiper_infer;
  Tensor<float> f1(1, 1, 4);
  Tensor<float> f2(4);
  ASSERT_EQ(f1.raw_shapes().size(), 3);
  ASSERT_EQ(f2.raw_shapes().size(), 1);

}

TEST(test_tensor, create_3dtensor) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  ASSERT_EQ(f1.shapes().size(), 3);
  ASSERT_EQ(f1.size(), 24);
}

TEST(test_tensor, get_infos) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  ASSERT_EQ(f1.channels(), 2);
  ASSERT_EQ(f1.rows(), 3);
  ASSERT_EQ(f1.cols(), 4);
}

TEST(test_tensor, tensor_init1D) {
  using namespace kuiper_infer;
  Tensor<float> f1(4);
  f1.Fill(1.f);
  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor1D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t size = raw_shapes.at(0);
  LOG(INFO) << "data numbers: " << size;
  f1.Show();
}

TEST(test_tensor, tensor_init2D) {
  using namespace kuiper_infer;
  Tensor<float> f1(4, 4);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor2D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_init3D_3) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 3-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t channels = raw_shapes.at(0);
  const uint32_t rows = raw_shapes.at(1);
  const uint32_t cols = raw_shapes.at(2);

  LOG(INFO) << "data channels: " << channels;
  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_init3D_2) {
  using namespace kuiper_infer;
  Tensor<float> f1(1, 2, 3);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 2-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_init3D_1) {
  using namespace kuiper_infer;
  Tensor<float> f1(1, 1, 3);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 1-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t size = raw_shapes.at(0);

  LOG(INFO) << "data numbers: " << size;
  f1.Show();
}

TEST(test_fill_reshape, fill1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  std::vector<float> values(2 * 3 * 4);
  // 将1到12填充到values中
  for (int i = 0; i < 24; ++i) {
    values.at(i) = float(i + 1);
  }
  f1.Fill(values);
  f1.Show();
}


// reshape

TEST(test_fill_reshape, reshape1) {
  using namespace kuiper_infer;
  LOG(INFO) << "-------------------Reshape-------------------";
  Tensor<float> f1(2, 3, 4);
  std::vector<float> values(2 * 3 * 4);
  // 将1到12填充到values中
  for (int i = 0; i < 24; ++i) {
    values.at(i) = float(i + 1);
  }
  f1.Fill(values);
  f1.Show();
  /// 将大小调整为(4, 3, 2)
  f1.Reshape({4, 3, 2}, true);
  LOG(INFO) << "-------------------After Reshape-------------------";
  f1.Show();
}

// size

TEST(test_tensor_size, tensor_size1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  LOG(INFO) << "-----------------------Tensor Get Size-----------------------";
  LOG(INFO) << "channels: " << f1.channels();
  LOG(INFO) << "rows: " << f1.rows();
  LOG(INFO) << "cols: " << f1.cols();
}
// values

TEST(test_tensor_values, tensor_values1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Rand();
  f1.Show();
  LOG(INFO) << "Data in the (1,1,1): " << f1.at(1, 1, 1);
}

// home work

TEST(test_homework, homework1_flatten1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Flatten(true);
  ASSERT_EQ(f1.raw_shapes().size(), 1);
  ASSERT_EQ(f1.raw_shapes().at(0), 24);
}

TEST(test_homework, homework1_flatten2) {
  using namespace kuiper_infer;
  Tensor<float> f1(12, 24);
  f1.Flatten(true);
  ASSERT_EQ(f1.raw_shapes().size(), 1);
  ASSERT_EQ(f1.raw_shapes().at(0), 24 * 12);
}

TEST(test_homework, homework2_padding1) {
  using namespace kuiper_infer;
  Tensor<float> tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);
  tensor.Padding({1, 2, 3, 4}, 0);
  ASSERT_EQ(tensor.rows(), 7);
  ASSERT_EQ(tensor.cols(), 12);

  int index = 0;
  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        if ((r >= 2 && r <= 4) && (c_ >= 3 && c_ <= 7)) {
          ASSERT_EQ(tensor.at(c, r, c_), 1.f) << c << " "
                                              << " " << r << " " << c_;
        }
        index += 1;
      }
    }
  }
}

TEST(test_homework, homework2_padding2) {
  using namespace kuiper_infer;
  ftensor tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);
  tensor.Padding({2, 2, 2, 2}, 3.14f);
  ASSERT_EQ(tensor.rows(), 8);
  ASSERT_EQ(tensor.cols(), 9);

  int index = 0;
  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        if (c_ <= 1 || r <= 1) {
          ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
        } else if (c >= tensor.cols() - 1 || r >= tensor.rows() - 1) {
          ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
        }
        if ((r >= 2 && r <= 5) && (c_ >= 2 && c_ <= 6)) {
          ASSERT_EQ(tensor.at(c, r, c_), 1.f);
        }
        index += 1;
      }
    }
  }
}

// transform

float MinusOne(float value) { return value - 1.f; }
TEST(test_transform, transform1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Rand();
  f1.Show();
  f1.Transform(MinusOne);
  f1.Show();
}