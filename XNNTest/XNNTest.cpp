
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
		ActivationLayer<Act>(1).Forward({ x }, out, nullptr);
		EXPECT_EQ(1, (int)out.size());
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
}

TEST(XNN, BatchNormalization) {
	XNNParams params(1);
	mt19937_64 mt;
	BatchNormalizationLayer layer(1);
	auto trainer = layer.CreateTrainer(params, mt);
	vector<float> in[3] = { { 3 }, { 4 }, { 5 } };
	vector<float> out[3];
	layer.Forward(in, out, 3, trainer.get());
	EXPECT_NEAR(-1.2f, out[0][0], 0.1f);
	EXPECT_NEAR(0.0f, out[1][0], 0.1f);
	EXPECT_NEAR(1.2f, out[2][0], 0.1f);
	vector<float> errorIn{1.3f}, errorOut;
	layer.Backward(*trainer, 0, in[0], out[0], errorOut, errorIn);
	EXPECT_NEAR(0.3f, errorOut[0], 0.1f);
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

TEST(XNN, ToStringFromString) {
	for (int i = 0; i < (int)stringTable.objectives.size(); i++)
		EXPECT_EQ(XNNObjective(i), XNNObjectiveFromString(ToString(XNNObjective(i))));
	for (int i = 0; i < (int)stringTable.activations.size(); i++)
		EXPECT_EQ(XNNActivation(i), XNNActivationFromString(ToString(XNNActivation(i))));
}

// TODO: もっとがんばる。
