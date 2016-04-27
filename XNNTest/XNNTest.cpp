
#include <gtest/gtest.h>
#include "../XNN/XNN.cpp"

using namespace XNN;

TEST(XNN, Sign) {
	EXPECT_EQ(-1, Sign(-3));
	EXPECT_EQ(+1, Sign(+3));
	EXPECT_EQ(0, Sign(0));
}

TEST(XNN, Adam) {
	// Adamの挙動の備忘録のようなもの
	AdamOptimizer optimizer(1, 1.0);
	optimizer.l1 = 0;
	optimizer.l2 = 0;
	EXPECT_NEAR(0.00100000, optimizer.GetStep(0, 1.0, 1), 0.000001);
	EXPECT_NEAR(0.00100000, optimizer.GetStep(0, 1.0, 2), 0.000001);
	EXPECT_NEAR(0.00100000, optimizer.GetStep(0, 1.0, 3), 0.000001);
	EXPECT_NEAR(0.00094827, optimizer.GetStep(0, 0.5, 4), 0.000001);
	EXPECT_NEAR(0.00096495, optimizer.GetStep(0, 1.5, 5), 0.000001);
}

namespace {
	template<XNNActivation Act>
	float activation(float x) {
		vector<float> out;
		ActivationLayer<Act>(1).Forward({ x }, out);
		EXPECT_EQ(1, out.size());
		return out[0];
	}
}
TEST(XNN, ActivationLayer) {
	EXPECT_EQ(0.0f, activation<XNNActivation::ReLU>(-3));
	EXPECT_EQ(3.0f, activation<XNNActivation::ReLU>(+3));
	EXPECT_EQ(-0.25f, activation<XNNActivation::PReLU>(-1)); // 重みが初期値の場合
	EXPECT_EQ(1.0f, activation<XNNActivation::PReLU>(+1));
	EXPECT_EQ(0.0f, activation<XNNActivation::Sigmoid>(-100));
	EXPECT_EQ(0.5f, activation<XNNActivation::Sigmoid>(0));
	EXPECT_EQ(1.0f, activation<XNNActivation::Sigmoid>(100));
	EXPECT_EQ(-3.0f, activation<XNNActivation::Identity>(-3));
	EXPECT_EQ(+3.0f, activation<XNNActivation::Identity>(+3));
}

TEST(XNN, SVNLight) {
	stringstream ss1, ss2;
	ss1 << "1.23 1:5.6 3:8" << endl;
	ss1 << "0,2 2:2 3:4.5678" << endl;
	auto data = LoadSVMLight(ss1, 4);
	ss2 << fixed << setprecision(1);
	SaveSVMLight(ss2, data);
	EXPECT_EQ(
		"1.2 1:5.6 2:0.0 3:8.0 4:0.0\n"
		"0.0,2.0 1:0.0 2:2.0 3:4.6 4:0.0\n", ss2.str());
}

// TODO: もっとがんばる。
