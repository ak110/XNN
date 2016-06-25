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
	vector<XNNData> LoadSVMLight(const string& path, int inUnits, int fMinIndex = 1);
	vector<XNNData> LoadSVMLight(istream& is, int inUnits, int fMinIndex = 1);
	// svmlight形式(若干独自拡張)の保存
	void SaveSVMLight(const string& path, const vector<XNNData>& data, int fMinIndex = 1);
	void SaveSVMLight(ostream& os, const vector<XNNData>& data, int fMinIndex = 1);

	// 目的関数の種類
	enum struct XNNObjective {
		// 線形回帰(二乗誤差損失)
		RegLinear,
		// ロジスティック回帰
		RegLogistic,
		// ロジスティック2クラス分類
		BinaryLogistic,
		// ソフトマックス多クラス分類
		MultiSoftmax,
	};

	// 活性化関数の種類
	enum class XNNActivation {
		ReLU,
		PReLU,
		ELU,
		Sigmoid, // 今のところ出力層専用
		Softmax,  // 今のところ出力層専用
	};

	// 文字列との相互変換
	string ToString(XNNObjective value);
	string ToString(XNNActivation value);
	XNNObjective XNNObjectiveFromString(const string& str);
	XNNActivation XNNActivationFromString(const string& str);

	// パラメータ
	struct XNNParams {
		// 目的関数
		XNNObjective objective = XNNObjective::RegLogistic;
		// 入力、隠れ層、出力のユニット数
		int inUnits, hiddenUnits = 32, outUnits = 1;
		// 隠れ層の数。1で3層パーセプトロン相当。0で隠れ層無し(＝線形モデル)になる。
		int hiddenLayers = 3;
		// 入力のスケーリングをするか否か
		bool scaleInput = true;
		// ミニバッチのサイズ
		int miniBatchSize = 100;
		// ミニバッチの最低回数。訓練データがこの値未満なら整数倍して超えるようにして学習する。
		int minMiniBatchCount = 1000;
		// 検証データのRMSEの最小値がこの回数だけ更新されなければ終了する。0なら初回で終了。
		int earlyStoppingTolerance = 3;
		// 学習の途中経過を多めに出すなら1。既定値も1。(データやネットワークが大きい時用)
		int verbose = 1;
		// 2クラス分類のときの正例(ラベルが1のデータ)の重み。
		float scalePosWeight = -1;
		// 隠れ層の活性化関数(ReLU or PReLU or ELU)
		XNNActivation activation = XNNActivation::ELU;
		// ペナルティの重み
		float l1 = 0, l2 = 0.01f;
		// Dropoutする場合に、残す割合を(0, 1]で指定する。1ならDropoutしない。0.75なら3/4は残す。
		float dropoutKeepProb = 1;
		// BatchNormalizationするか否か
		bool batchNormalization = false;
		// 学習を終了する最小・最大のepoch数
		int minEpoch = 1, maxEpoch = 10;
		// 初期化
		XNNParams(int inUnits) : inUnits(inUnits) {}
		XNNParams(int inUnits, int hiddenUnits, int outUnits, int hiddenLayers)
			: inUnits(inUnits), hiddenUnits(hiddenUnits), outUnits(outUnits), hiddenLayers(hiddenLayers) {}
	};

	// 学習器
	class XNNModel {
		struct XNNImpl;
		unique_ptr<XNNImpl> impl;
	public:
		XNNModel(const XNNParams& params);
		XNNModel(const string& path);
		~XNNModel();

		// ログファイルを設定する
		void SetLog(ostream& os);
		// 学習時の損失の変化を出力するCSVファイルを設定する
		void SetHistory(ostream& os);

		// モデルの保存
		void Save(const string& path) const;
		// モデルの読み込み
		void Load(const string& path);
		// 文字列化
		string Dump() const;
		// 学習
		void Train(vector<XNNData>&& trainData, vector<XNNData>&& testData);
		// 評価(集計結果をcoutに出して、生の値をstringで返す。)
		string Predict(vector<XNNData>&& testData) const;
		// 評価(スレッドセーフ、非並列)
		vector<float> Predict(vector<float>&& in) const;
		// 特徴の重要度のようなものを返す。(単に入力層の重みのRMSなので厳密ではない)
		vector<float> GetFScore() const;
	};
}
