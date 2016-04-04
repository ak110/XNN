#pragma once
#include <memory>
#include <vector>
#include <string>

namespace XNN {
	using namespace std;

	// 例外
	struct XNNException : public exception {
		string msg;
		XNNException(const string& msg) : msg(msg) {}
		const char* what() const noexcept override {
			return msg.c_str();
		}
	};

	// 学習器が使うラベル付きデータ
	struct XNNData {
		vector<float> in, out;
	};

	// svmlight形式(若干独自拡張)を読み込む
	vector<XNNData> LoadSVMLight(const string& path, int inUnits);
	vector<XNNData> LoadSVMLight(istream& is, int inUnits);
	// svmlight形式(若干独自拡張)の保存
	void SaveSVMLight(const string& path, const vector<XNNData>& data);
	void SaveSVMLight(ostream& os, const vector<XNNData>& data);

	// 目的関数の種類
	enum struct XNNObjective {
		// ロジスティック回帰
		RegLogistic,
		// ロジスティック2クラス分類
		BinaryLogistic,
		// ソフトマックス多クラス分類
		MultiSoftmax,
	};
#pragma pack(push, 1)
	// パラメータ
	struct XNNParams {
		// 目的関数
		XNNObjective objective = XNNObjective::RegLogistic;
		// 入力、隠れ層、出力のユニット数
		int32_t inUnits, hiddenUnits, outUnits;
		// 隠れ層の数。1で3層パーセプトロン相当。0以下は未対応。
		int32_t hiddenLayers;
		// 入力のスケーリングをするなら1、しないなら0
		int32_t scaleInput = 1;
		// ミニバッチのサイズ
		int32_t miniBatchSize = 100;
		// ミニバッチの最低回数。訓練データがこの値未満なら整数倍して超えるようにして学習する。
		int32_t minMiniBatchCount = 1000;
		// 検証データのRMSEの差がこの値未満になったら学習終了。(ただし最低でも訓練データを1回は使用する。)
		double stopDeltaRMSE = 0.0005; // 0.05%
		// 学習の途中経過を多めに出すなら1。既定値も1。(データやネットワークが大きい時用)
		int32_t verbose = 1;
		// 念のため互換性用
		int32_t reserved[65];
		// 初期化
		XNNParams(int inUnits, int hiddenUnits, int outUnits, int hiddenLayers)
			: inUnits(inUnits), hiddenUnits(hiddenUnits), outUnits(outUnits), hiddenLayers(hiddenLayers) {}
	};
#pragma pack(pop)
	static_assert(sizeof(XNNParams) == 304, "sizeof(XNNParams)");

	// 学習器
	class XNNModel {
		struct XNNImpl;
		unique_ptr<XNNImpl> impl;
	public:
		XNNModel(const XNNParams& params);
		XNNModel(const string& path);
		~XNNModel();

		struct PredictResult { string statistics, raw; };

		// モデルの保存
		void Save(const string& path) const;
		// モデルの読み込み
		void Load(const string& path);
		// 学習
		void Train(vector<XNNData>&& trainData, vector<XNNData>&& testData);
		// 評価して文字列化
		PredictResult Predict(vector<XNNData>&& testData) const;
		// 評価(スレッドセーフ、非並列)
		vector<float> Predict(vector<float>&& in) const;
	};
}
