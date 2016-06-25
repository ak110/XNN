
#include <gtest/gtest.h>
#define main ___main
#include "../XNN/Main.cpp"
#undef main

int main(int argc, char* argv[]) {
	::testing::GTEST_FLAG(break_on_failure) = true;   // JITデバッグするため(既定ではテスト失敗になる)
	::testing::GTEST_FLAG(catch_exceptions) = false;  // JITデバッグするため(既定ではテスト失敗になる)
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

// and, or, xorを学習してみる
TEST(Main, Heavy_AndOrXor) {
	// データを作る
	vector<XNNData> data;
	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			XNNData d;
			d.in.push_back((float)x);
			d.in.push_back((float)y);
			d.out.push_back((float)(x & y));
			d.out.push_back((float)(x | y));
			d.out.push_back((float)(x ^ y));
			data.emplace_back(move(d));
		}
	}

	XNNParams params(2, 8, 3, 1);
	params.objective = XNNObjective::BinaryLogistic;
	XNNModel dnn(params);
	dnn.Train(vector<XNNData>(data), vector<XNNData>(data));

	auto result00 = dnn.Predict(vector<float>(data[0].in));
	auto result01 = dnn.Predict(vector<float>(data[1].in));
	auto result10 = dnn.Predict(vector<float>(data[2].in));
	auto result11 = dnn.Predict(vector<float>(data[3].in));
	// and
	EXPECT_LT(result00[0], 0.5f);
	EXPECT_LT(result01[0], 0.5f);
	EXPECT_LT(result10[0], 0.5f);
	EXPECT_GT(result11[0], 0.5f);
	// or
	EXPECT_LT(result00[1], 0.5f);
	EXPECT_GT(result01[1], 0.5f);
	EXPECT_GT(result10[1], 0.5f);
	EXPECT_GT(result11[1], 0.5f);
	// xor
	EXPECT_LT(result00[2], 0.5f);
	EXPECT_GT(result01[2], 0.5f);
	EXPECT_GT(result10[2], 0.5f);
	EXPECT_LT(result11[2], 0.5f);

	// Dump/Save/Loadもついでにテスト
	auto d = dnn.Dump();
	dnn.Save("test.model.tmp");
	XNNModel dnn2("test.model.tmp");
	EXPECT_EQ(d, dnn2.Dump());
}

// 過去のバージョンで保存したモデルが正しく読み込めることを確認
TEST(Main, OldModel) {
	XNNModel dnn("../XNNTest/OldModel.model");
	auto result00 = dnn.Predict(vector<float>({ 0, 0 }));
	auto result01 = dnn.Predict(vector<float>({ 0, 1 }));
	auto result10 = dnn.Predict(vector<float>({ 1, 0 }));
	auto result11 = dnn.Predict(vector<float>({ 1, 1 }));
	// and
	EXPECT_NEAR(result00[0], 0.005228f, 0.000001f);
	EXPECT_NEAR(result01[0], 0.002399f, 0.000001f);
	EXPECT_NEAR(result10[0], 0.002471f, 0.000001f);
	EXPECT_NEAR(result11[0], 0.997612f, 0.000001f);
	// or
	EXPECT_NEAR(result00[1], 0.012223f, 0.000001f);
	EXPECT_NEAR(result01[1], 0.998801f, 0.000001f);
	EXPECT_NEAR(result10[1], 0.999242f, 0.000001f);
	EXPECT_NEAR(result11[1], 1.000000f, 0.000001f);
	// xor
	EXPECT_NEAR(result00[2], 0.008208f, 0.000001f);
	EXPECT_NEAR(result01[2], 0.997024f, 0.000001f);
	EXPECT_NEAR(result10[2], 0.996736f, 0.000001f);
	EXPECT_NEAR(result11[2], 0.003302f, 0.000001f);
}
