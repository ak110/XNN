
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

TEST(XNN, ToStringFromString) {
	for (int i = 0; i < (int)stringTable.objectives.size(); i++)
		EXPECT_EQ(XNNObjective(i), XNNObjectiveFromString(ToString(XNNObjective(i))));
	for (int i = 0; i < (int)stringTable.activations.size(); i++)
		EXPECT_EQ(XNNActivation(i), XNNActivationFromString(ToString(XNNActivation(i))));
}

TEST(Formatter, ProbabilityFormatter) {
	EXPECT_EQ("100.0%", ProbabilityFormatter()(1.0));
	EXPECT_EQ(" 0.0%", ProbabilityFormatter()(0.0));
	EXPECT_EQ(" 0.1%", ProbabilityFormatter()(0.00099));
	EXPECT_EQ("12.3%", ProbabilityFormatter()(0.1234));
}

TEST(Formatter, PlainValueFormatter) {
	EXPECT_EQ("       1", PlainValueFormatter()(1.0));
	EXPECT_EQ("   0.123", PlainValueFormatter()(0.1234));
	EXPECT_EQ("1.23e+03", PlainValueFormatter()(1234));
}

namespace {
	// 各レイヤーが基本的に満たすべき制約。出来るだけ言語仕様で縛りたいが、
	// けちくさい高速化などのために変な仕組みになっているところも多いので…。
	void TestLayerBasics(ILayer&& layer) {
		vector<XNNData> data{
			XNNData{ { 1, 2, 3 }, { 4, 5, 6 } },
		};
		XNNParams params(1);
		mt19937_64 mt;

		// Initializeの確認
		double outputNorm = NAN;
		layer.Initialize(data, mt, 33.0, outputNorm);
		EXPECT_TRUE(!std::isnan(outputNorm) && !std::isinf(outputNorm)); // outputNormを何かしら設定することの確認

		auto trainer = layer.CreateTrainer(params, mt);
		EXPECT_NE(nullptr, trainer.get());
		trainer->Clear(); // とりあえず死なないことだけでも。

		// 順伝搬
		vector<float> in[3] = { { 3, 3, 3 }, { 4, 4, 4 }, { 5, 5, 5 } };
		vector<float> out[3];
		layer.Forward(in, out, 3, trainer.get());
		EXPECT_EQ((size_t)3, out[0].size());
		EXPECT_EQ((size_t)3, out[1].size());
		EXPECT_EQ((size_t)3, out[2].size());

		// 逆伝搬
		vector<float> errorIn{ 1.3f, 1.3f, 1.3f }, errorOut;
		layer.Backward(*trainer, 0, in[0], out[0], errorOut, errorIn);
		EXPECT_GE(errorOut.size(), (size_t)1); // 出力が1個以上あることの確認

		// 更新
		trainer->Update(); // とりあえず死なないことだけでも。

		// セーブロード (とりあえず死なないことだけでも。)
		stringstream ss;
		layer.Save(ss);
		layer.Load(ss);
	}
}
TEST(Layers, Basics_InputScaling) { TestLayerBasics(InputScalingLayer(3)); }
TEST(Layers, Basics_OutputScaling) { TestLayerBasics(OutputScalingLayer(3)); }
TEST(Layers, Basics_FullyConnected) { TestLayerBasics(FullyConnectedLayer(3, 3)); }
TEST(Layers, Basics_ReLU) { TestLayerBasics(ActivationLayer<XNNActivation::ReLU>(3)); }
TEST(Layers, Basics_PReLU) { TestLayerBasics(ActivationLayer<XNNActivation::PReLU>(3)); }
TEST(Layers, Basics_ELU) { TestLayerBasics(ActivationLayer<XNNActivation::ELU>(3)); }
TEST(Layers, Basics_Sigmoid) { TestLayerBasics(ActivationLayer<XNNActivation::Sigmoid>(3)); }
TEST(Layers, Basics_Softmax) { TestLayerBasics(ActivationLayer<XNNActivation::Softmax>(3)); }
TEST(Layers, Basics_BatchNormalization) { TestLayerBasics(BatchNormalizationLayer(3)); }
TEST(Layers, Basics_Dropout) { TestLayerBasics(DropoutLayer(3, 0.5)); }

namespace {
	template<XNNActivation Act>
	float Activate(float x) {
		vector<float> out;
		ActivationLayer<Act>(1).Forward({ x }, out, nullptr);
		EXPECT_EQ(1, (int)out.size());
		return out[0];
	}
}
TEST(Layers, ActivationLayer) {
	EXPECT_EQ(0.0f, Activate<XNNActivation::ReLU>(-3));
	EXPECT_EQ(3.0f, Activate<XNNActivation::ReLU>(+3));
	EXPECT_EQ(-0.25f, Activate<XNNActivation::PReLU>(-1)); // 重みが初期値の場合
	EXPECT_EQ(1.0f, Activate<XNNActivation::PReLU>(+1));
	EXPECT_EQ(0.0f, Activate<XNNActivation::Sigmoid>(-100));
	EXPECT_EQ(0.5f, Activate<XNNActivation::Sigmoid>(0));
	EXPECT_EQ(1.0f, Activate<XNNActivation::Sigmoid>(100));
}

TEST(Layers, BatchNormalization) {
	XNNParams params(1);
	mt19937_64 mt;
	BatchNormalizationLayer layer(1);
	auto trainer = layer.CreateTrainer(params, mt);
	// 順伝搬
	vector<float> in[3] = { { 3 }, { 4 }, { 5 } };
	vector<float> out[3];
	layer.Forward(in, out, 3, trainer.get());
	EXPECT_NEAR(-1.224744f, out[0][0], 0.000001f);
	EXPECT_NEAR(+0.000000f, out[1][0], 0.000001f);
	EXPECT_NEAR(+1.224744f, out[2][0], 0.000001f);
	// 逆伝搬
	vector<float> errorIn{1.3f}, errorOut;
	layer.Backward(*trainer, 0, in[0], out[0], errorOut, errorIn);
	EXPECT_NEAR(0.53f, errorOut[0], 0.1f);
	// 更新して再度順伝搬
	trainer->Update();
	layer.Forward(in, out, 3, trainer.get());
	EXPECT_NEAR(-1.222519f, out[0][0], 0.000001f);
	EXPECT_NEAR(+0.001000f, out[1][0], 0.000001f);
	EXPECT_NEAR(+1.224519f, out[2][0], 0.000001f);
}

TEST(SVNLight, LoadSave) {
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
