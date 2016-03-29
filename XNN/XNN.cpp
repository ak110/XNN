#include "XNN.h"
#include <omp.h>
#include <atomic>
#include <algorithm>
#include <random>
#include <cassert>
#include <mutex>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <numeric>
#include <memory>
#include <string>
#include <array>
#include <iostream>
#include <iomanip>
#include <cerrno>

#pragma warning(disable : 4127) // 条件式が定数です。
#pragma warning(disable : 4100) // 引数は関数の本体部で 1 度も参照されません。

namespace XNN {
	using namespace std::chrono;

	// 雑なoperator色々
	template<class Range, class T>
	void operator*=(Range& x, T y) { for (auto& r : x) r *= y; }
	template<class Range, class T>
	void operator/=(Range& x, T y) { for (auto& r : x) r /= y; }

	// 密ベクトル。基本はvector<float>でいいのだが、SparseVectorとインターフェースを似せるため用意。
	template<class ValueType = float>
	struct DenseVector : public vector<ValueType> {
		typedef vector<ValueType> BaseType;
		explicit DenseVector(size_t n) { this->resize(n, ValueType()); }
		void Clear() { fill(BaseType::begin(), BaseType::end(), ValueType()); }
		void operator+=(const pair<size_t, ValueType>& p) {
#pragma omp atomic
			(*this)[p.first] += p.second; // resizeは事前にしてある前提とする。
		}
		void operator*=(double factor) {
			for (auto& p : (vector<ValueType>&)*this) p *= ValueType(factor);
		}
		double GetL1Norm() const {
			double s = 0.0;
			for (const auto& p : *this) s += abs(p.second);
			return s;
		}
		double GetL2Norm() const {
			double s = 0.0;
			for (const auto& p : *this) s += p.second * p.second;
			return sqrt(s);
		}
		struct const_iterator {
			const DenseVector<ValueType>* parent;
			size_t i;
			void operator++() { i++; }
			const_iterator& operator++(int) { ++i; return *this; }
			pair<size_t, ValueType> operator*() const { return make_pair(i, (*parent)[i]); }
			bool operator==(const const_iterator& other) const { return i == other.i; }
			bool operator!=(const const_iterator& other) const { return i != other.i; }
		};
		const_iterator begin() const { return const_iterator{ this, 0 }; }
		const_iterator end() const { return const_iterator{ this, this->size() }; }
	};

	// 符号を返す
	template<class T>
	constexpr int Sign(T value) { return T() < value ? +1 : value < T() ? -1 : 0; }

	// 学習の進行状況を表示するためのクラス
	struct ProgressTimer {
		uint64_t cur = 0, size = 1;
		high_resolution_clock::time_point start = high_resolution_clock::now();
		void Set(uint64_t cur_, uint64_t size_) {
			assert(1 <= cur_ && cur_ <= size_);
			cur = cur_;
			size = size_;
		}
		string ToStringCount() const {
			stringstream ss;
			ss << "count=" << setw((int)log10(size) + 1) << cur << "/" << size;
			return ss.str();
		}
		string ToStringTime() const {
			auto time = duration<double>(high_resolution_clock::now() - start).count();
			auto remain = (int)ceil((size - cur) * time / cur);
			stringstream ss;
			ss << "time=" << (int)round(time) << "秒"
				<< " remain=";
			struct Unit { string name; int seconds; };
			for (auto x : {
				Unit{ "日", 24 * 60 * 60 },
				Unit{ "時間", 60 * 60 },
				Unit{ "分", 60 },
				Unit{ "秒", 1 },
			}) {
				if (x.seconds <= remain || x.seconds == 1) {
					auto t = remain / x.seconds;
					ss << t << x.name;
					remain -= t * x.seconds;
				}
			}
			return ss.str();
		}
	};

	struct IScore {
		virtual void Add(float label, float pred) = 0;
		virtual void Add(const vector<float>& label, const vector<float>& pred) = 0;
		virtual string ToString() const = 0;
	};
	// 二乗平均平方根を算出するためのクラス(スレッドセーフ)
	struct RegressionScore : public IScore {
		mutex mtx;
		struct {
			double totalA = 0.0, total2 = 0.0;
			size_t count = 0;
		} values;
		struct {
			int precision = 1;
			int width = 3;
			double scale = 100;
			string suffix = "%";
		} formats;
		RegressionScore() = default;
		RegressionScore(const RegressionScore& other) {
			values = other.values;
			formats = other.formats;
		}
		void Clear() { values = remove_reference<decltype(values)>::type(); }
		void operator+=(const RegressionScore& other) {
			lock_guard<mutex> lock(mtx);
			values.totalA += other.values.totalA;
			values.total2 += other.values.total2;
			values.count += other.values.count;
		}
		void Add(double error) {
			lock_guard<mutex> lock(mtx);
			values.totalA += abs(error);
			values.total2 += error * error;
			values.count++;
		}
		// MAE(Mean absolete error、平均絶対誤差)
		double GetMAE() const { return values.totalA / values.count; }
		// RMSE(Root mean square error、二乗平均平方根誤差)
		double GetRMSE() const { return sqrt(values.total2 / values.count); }
		// 追加(二値分類/回帰用)
		void Add(float label, float pred) override {
			Add(label - pred);
		}
		// 追加(多クラス分類用)
		void Add(const vector<float>& label, const vector<float>& pred) override {
			assert(false);
		}
		// 文字列化
		string ToString() const override {
			stringstream ss;
			ss << fixed << setprecision(formats.precision)
				<< "MAE=" << setw(formats.width + formats.precision) << GetMAE() * formats.scale << formats.suffix
				<< " RMSE=" << setw(formats.width + formats.precision) << GetRMSE() * formats.scale << formats.suffix;
			return ss.str();
		}
	};
	// 適合率、再現率、F値を計算する(スレッドセーフ)
	// sklearnのclassification_reportを参考にしてみた。
	struct ClassificationScore : IScore {
		struct CountList : array<atomic<uint64_t>, 3> {
			CountList() = default;
			CountList(const CountList& other) {
				for (size_t i = 0; i < size(); i++)
					(*this)[i] = other[i].load();
			}
		};
		vector<CountList> list;
		ClassificationScore(size_t classCount = 2) : list(classCount) {}
		ClassificationScore(const ClassificationScore& other) { *this = other; }
		void Clear() { *this = ClassificationScore(list.size()); }
		void operator=(const ClassificationScore& other) {
			assert(list.size() == other.list.size());
			for (size_t i = 0; i < list.size(); i++)
				for (size_t j = 0; j < list[i].size(); j++)
					list[i][j] = other.list[i][j].load();
		}
		void operator+=(const ClassificationScore& other) {
			assert(list.size() == other.list.size());
			for (size_t i = 0; i < list.size(); i++)
				for (size_t j = 0; j < list[i].size(); j++)
					list[i][j] += other.list[i][j];
		}
		// 追加(多クラス分類用)
		void Add(const vector<float>& label, const vector<float>& pred) override {
			assert(label.size() == list.size());
			assert(pred.size() == list.size());
			size_t labelClass = max_element(label.begin(), label.end()) - label.begin();
			size_t pickClass = max_element(pred.begin(), pred.end()) - pred.begin();
			Add(labelClass, pickClass);
		}
		// 追加(二値分類/回帰用)
		void Add(float label, float pred) override {
			Add(size_t(0.5 < label), size_t(0.5 < pred));
		}
		void Add(size_t labelClass, size_t pickClass) {
			if (labelClass == pickClass) {
				list[labelClass][0]++;
			} else {
				list[labelClass][1]++;
				list[pickClass][2]++;
			}
		}
		// 抽出回数
		uint64_t GetPickCount(size_t classIndex) const {
			return list[classIndex][0] + list[classIndex][2];
		}
		// 正解回数
		uint64_t GetLabelCount(size_t classIndex) const {
			return list[classIndex][0] + list[classIndex][1];
		}
		// 適合率 (抽出したものが正しかった率)
		double GetPrecision(size_t classIndex) const {
			return double(list[classIndex][0]) / GetPickCount(classIndex);
		}
		// 再現率 (正しい物をどれだけ抽出出来たか率)
		double GetRecall(size_t classIndex) const {
			return double(list[classIndex][0]) / GetLabelCount(classIndex);
		}
		// F値
		double GetFValue(size_t classIndex) const {
			auto p = GetPrecision(classIndex), r = GetRecall(classIndex);
			return 2 * p * r / (p + r);
		}
		// 文字列化
		string ToString() const override {
			uint64_t total = 0;
			for (auto& l : list)
				total += l[0] + l[1];
			int indexWidth = (int)floor(log10(list.size() - 1)) + 1;
			stringstream ss;
			ss << fixed << setprecision(1);
			double avgPrec = 0, avgRecl = 0, avgFval = 0;
			for (size_t classIndex = 0; classIndex < list.size(); classIndex++) {
				auto prec = GetPrecision(classIndex);
				auto recl = GetRecall(classIndex);
				auto fval = GetFValue(classIndex);
				auto pick = (double)GetPickCount(classIndex) / total;
				ss << "class[" << setw(indexWidth) << classIndex << "]:"
					<< " 適合率=" << setw(4) << prec * 100 << "%"
					<< " 再現率=" << setw(4) << recl * 100 << "%"
					<< " F値=" << setw(4) << fval * 100 << "%"
					<< " 選択率=" << setw(4) << pick * 100 << "%"
					<< endl;
				avgPrec += prec * pick;
				avgRecl += recl * pick;
				avgFval += fval * pick;
			}
			ss << "average:" << setw(indexWidth) << ""
				<< " 適合率=" << setw(4) << avgPrec * 100 << "%"
				<< " 再現率=" << setw(4) << avgRecl * 100 << "%"
				<< " F値=" << setw(4) << avgFval * 100 << "%"
				<< endl;
			return ss.str();
		}
	};

	// パラメータを更新する人たちの共通的な処理
	template<class DerivedClass>
	struct OptimizerBase {
		const size_t dimension;
		// 重みのスケール。大きくすると、学習率が大きくL2が小さくなる。
		double scale;
		// ペナルティ
		double l1 = 0.001, l2 = 0.01;
		// 更新回数のカウンタ
		uint64_t t;

		OptimizerBase(size_t dimension, double scale) : dimension(dimension), scale(scale), t(0ull) {}

		template<class TableType, class GradListType>
		uint64_t Update(TableType& table, const GradListType& gradList) {
			const auto lt = ++t;
			const auto scale_ = this->scale, l1_ = this->l1, l2_ = this->l2;
			const auto x = begin(table);
			for (const auto& p : gradList) {
				if (abs(p.second) < 1e-10) continue; // 勾配がほぼ0のものはskip
				auto i = p.first;
				assert(i < dimension);
				// 更新
				auto oldX = x[i];
				auto g = p.second - Sign(oldX) * l1_ - oldX * l2_ / scale_;
				auto baseStep = static_cast<DerivedClass*>(this)->GetStep(i, g, lt);
				auto step = baseStep * scale_;
				auto newX = oldX + static_cast<typename remove_reference<decltype(oldX)>::type>(step);
				if ((oldX < 0 && 0 < newX) || (newX < 0 && 0 < oldX))
					newX = 0;
				x[i] = newX;
			}
			return lt;
		}
	};

	// Adamによるパラメータの更新
	struct AdamOptimizer : public OptimizerBase<AdamOptimizer> {
		unique_ptr<double[]> m;
		unique_ptr<double[]> v;
		double alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;

		AdamOptimizer(size_t dimension, double scale) : OptimizerBase(dimension, scale), m(new double[dimension]), v(new double[dimension]) {
			fill(m.get(), m.get() + dimension, 0.0);
			fill(v.get(), v.get() + dimension, 0.0);
		}

		double GetStep(size_t i, double g, uint64_t lt) {
			m[i] = beta1 * m[i] + (1.0 - beta1) * g;
			v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
			auto hm = m[i] / (1.0 - pow(beta1, lt));
			auto hv = v[i] / (1.0 - pow(beta2, lt));
			return alpha * hm / (sqrt(hv) + epsilon);
		}
	};

	// 活性化関数
	enum class ActivationFunction { ReLU, Sigmoid, Identity };
	template<ActivationFunction Act> float activation(float x);
	template<> float activation<ActivationFunction::ReLU>(float x) { return max(0.f, x); }
	template<> float activation<ActivationFunction::Sigmoid>(float x) { return 1.0f / (1.0f + exp(-x)); }
	template<> float activation<ActivationFunction::Identity>(float x) { return x; }

	// 層を学習するクラス
	struct ILayerTrainer {
		virtual ~ILayerTrainer() {}
		// 勾配をクリアする
		virtual void Clear() = 0;
		// 勾配に従って重みを更新
		virtual void Update() = 0;
	};
	struct NullLayerTrainer : public ILayerTrainer {
		void Clear() override {}
		void Update() override {}
	};

	// 層
	struct ILayer {
		virtual ~ILayer() {}
		// モデルの読み込み
		virtual void Load(istream& s) = 0;
		// モデルの保存
		virtual void Save(ostream& s) const = 0;
		// 学習用に初期化
		// 学習率を程よくするために、入力のL1ノルムの平均(の推定値)を受け取り、出力のL1ノルムの平均(の推定値)を返す。(順に伝搬させる)
		virtual void Initialize(const vector<XNNData>& data, mt19937_64& rnd, double inputNorm, double& outputNorm) = 0;
		// 学習するクラスを作る
		virtual unique_ptr<ILayerTrainer> CreateTrainer() = 0;
		// 順伝搬
		virtual void Forward(const vector<float>& in, vector<float>& out) const = 0;
		// 逆伝搬
		virtual void Backward(
			ILayerTrainer& trainer,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const = 0;
	};

	// 入力をスケーリングする層
	struct InputScalingLayer : public ILayer {
		uint64_t inUnits;
		vector<float> scale;
		InputScalingLayer(uint64_t inUnits, float minScale = 1.0f) : inUnits(inUnits), scale(inUnits, minScale) {}
		// モデルの読み込み
		void Load(istream& s) {
			s.read((char*)&inUnits, sizeof inUnits);
			scale.resize(inUnits);
			s.read((char*)&scale[0], scale.size() * sizeof scale[0]);
		}
		// モデルの保存
		void Save(ostream& s) const {
			s.write((const char*)&inUnits, sizeof inUnits);
			s.write((const char*)&scale[0], scale.size() * sizeof scale[0]);
		}
		// 学習用に初期化
		// 学習率を程よくするために、入力のL1ノルムの平均(の推定値)を受け取り、出力のL1ノルムの平均(の推定値)を返す。(順に伝搬させる)
		void Initialize(const vector<XNNData>& data, mt19937_64& rnd, double inputNorm, double& outputNorm) {
			// スケールの決定。とりあえず絶対値の最大値で割り算する感じにする。
			for (size_t c = 0; c < data.size(); c++) {
				for (size_t i = 0; i < inUnits; i++) {
					auto a = abs(data[c].in[i]);
					if (scale[i] < a)
						scale[i] = a;
				}
			}
			for (size_t i = 0; i < inUnits; i++)
				scale[i] = 1.0f / scale[i];
			// 出力のL1ノルム
			outputNorm = 0;
			for (size_t c = 0; c < data.size(); c++) {
				for (size_t i = 0; i < inUnits; i++)
					outputNorm += abs(data[c].in[i] * scale[i]);
			}
			outputNorm /= data.size();
		}
		// 学習するクラスを作る
		unique_ptr<ILayerTrainer> CreateTrainer() {
			return unique_ptr<ILayerTrainer>(new NullLayerTrainer());
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out) const {
			out = in;
			for (size_t i = 0; i < inUnits; i++)
				out[i] *= scale[i];
		}
		// 逆伝搬
		void Backward(
			ILayerTrainer& trainer,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const {
			errorOut = errorIn;
		}
	};

	// 全層結合層
	// 結合と活性化関数はそれぞれ層にするのが今時っぽいけど、
	// ReLUに特化してちょっと計算をサボるためにセットで層にする。
	template<ActivationFunction Act> struct FullyConnectedLayer : public ILayer {
		uint64_t inUnits, outUnits;
		double inputNorm;
		vector<float> weights;
		vector<float> biases;
		FullyConnectedLayer(uint64_t inUnits, uint64_t outUnits) : inUnits(inUnits), outUnits(outUnits) {
			weights.resize(inUnits * outUnits);
			biases.resize(outUnits);
		}
		// モデルの読み込み
		void Load(istream& s) override {
			s.read((char*)&inUnits, sizeof inUnits);
			s.read((char*)&outUnits, sizeof outUnits);
			weights.resize(inUnits * outUnits);
			biases.resize(outUnits);
			s.read((char*)&weights[0], weights.size() * sizeof weights[0]);
			s.read((char*)&biases[0], biases.size() * sizeof biases[0]);
		}
		// モデルの保存
		void Save(ostream& s) const override {
			s.write((const char*)&inUnits, sizeof inUnits);
			s.write((const char*)&outUnits, sizeof outUnits);
			s.write((const char*)&weights[0], weights.size() * sizeof weights[0]);
			s.write((const char*)&biases[0], biases.size() * sizeof biases[0]);
		}
		// 学習用に初期化
		void Initialize(const vector<XNNData>&, mt19937_64& rnd, double inputNorm_, double& outputNorm) override {
			// 分散の合計が1の一様分布がいいという噂。http://deeplearning.net/tutorial/mlp.html
			// sigmoidの時は更に4倍
			inputNorm = inputNorm_;
			auto s = sqrt(3.0f / (float(inputNorm * 2)));
			if (Act == ActivationFunction::Sigmoid)
				s *= 4;
			auto nd = uniform_real_distribution<float>(-s, s);
			for (auto& x : weights)
				x = nd(rnd);
			for (auto& x : biases)
				x = 0.1f / inUnits; // バイアスは適当な正の定数。dead neurons対策。
			outputNorm = outUnits / 2.0; // ReLU1個やSigmoid1個あたり平均0.5と見なす。(実際の挙動はよく知らない…)
		}
		// 学習するクラス
		struct Trainer : ILayerTrainer {
			FullyConnectedLayer<Act>& owner;
			AdamOptimizer optimizerW, optimizerB;
			DenseVector<> gradW, gradB;
			Trainer(FullyConnectedLayer<Act>& owner) :
				owner(owner),
				// 入力の大きさに応じて謎チューニング
				optimizerW(owner.weights.size(), 10.0 / sqrt(owner.inputNorm)),
				optimizerB(owner.biases.size(), 10.0 / sqrt(owner.inputNorm)),
				gradW(owner.weights.size()),
				gradB(owner.biases.size()) {
				// バイアスは正則化項を入れない
				optimizerB.l1 = 0;
				optimizerB.l2 = 0;
			}
			void Clear() override {
				gradW.Clear();
				gradB.Clear();
			}
			void Update() override {
				optimizerW.Update(owner.weights, gradW);
				optimizerB.Update(owner.biases, gradB);
			}
		};
		// 学習するクラスを作る
		unique_ptr<ILayerTrainer> CreateTrainer() override {
			return unique_ptr<ILayerTrainer>(new Trainer(*this));
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out) const override {
			assert(in.size() == inUnits);
			out.clear();
			out.resize(outUnits, 0.0f);
			for (size_t o = 0; o < outUnits; o++) {
				for (size_t i = 0; i < inUnits; i++)
					out[o] += in[i] * weights[o * inUnits + i];
			}
			for (size_t o = 0; o < outUnits; o++)
				out[o] = activation<Act>(out[o] + biases[o]);
		}
		// 逆伝搬
		void Backward(ILayerTrainer& trainer_,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const override {
			assert(in.size() == inUnits);
			assert(out.size() == outUnits);
			assert(errorIn.size() == outUnits);
			errorOut.clear();
			errorOut.resize(inUnits, 0.0f);
			auto& trainer = (Trainer&)trainer_;
			for (size_t o = 0; o < outUnits; o++) {
				if (Act == ActivationFunction::ReLU && out[o] <= FLT_EPSILON)
					continue; // サボる
				if (abs(errorIn[o]) <= FLT_EPSILON)
					continue;
				// 誤差の伝播
				for (size_t i = 0; i < inUnits; i++)
					errorOut[i] += errorIn[o] * weights[o * inUnits + i];
				// wの勾配
				for (size_t i = 0; i < inUnits; i++)
					trainer.gradW += make_pair(o * inUnits + i, errorIn[o] * in[i]);
				// bの勾配
				trainer.gradB += make_pair(o, errorIn[o]);
			}
		}
	};
	// 多クラス分類用のSoftmax関数
	struct SoftmaxLayer : public ILayer {
		// モデルの読み込み
		void Load(istream& s) override {}
		// モデルの保存
		void Save(ostream& s) const override {}
		// 学習用に初期化
		void Initialize(const vector<XNNData>& data, mt19937_64& rnd, double inputNorm, double& outputNorm) override {
			outputNorm = 1.0;
		}
		// 学習するクラスを作る
		unique_ptr<ILayerTrainer> CreateTrainer() override {
			return unique_ptr<ILayerTrainer>(new NullLayerTrainer());
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out) const override {
			out = in;
			auto m = *max_element(out.begin(), out.end());
			for (auto& x : out)
				x = exp(x - m); // 最大値を引く(オーバーフロー対策)
			auto s = accumulate(out.begin(), out.end(), 0.0f);
			out /= s;
		}
		// 逆伝搬
		void Backward(
			ILayerTrainer& trainer,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const override {
			errorOut = errorIn;
		}
	};

	// 実装。
	struct XNNModel::XNNImpl {
		XNNParams params;
		vector<unique_ptr<ILayer>> layers;
		vector<unique_ptr<ILayerTrainer>> trainers;
		mt19937_64 mt;

		XNNImpl(const XNNParams& params) : params(params) {
			Initialize();
		}
		XNNImpl(const string& path) : params(0, 0, 0, 0) { Load(path); }

		void Save(const string& path) const {
			ofstream fs(path, ios_base::binary);
			fs.write((const char*)&params, sizeof params);
			for (auto& l : layers)
				l->Save(fs);
		}
		void Load(const string& path) {
			ifstream fs(path, ios_base::binary);
			fs.read((char*)&params, sizeof params);
			Initialize();
			for (auto& l : layers)
				l->Load(fs);
		}
		void Initialize() {
			// パラメータのチェック
			if (params.inUnits < 1)
				throw XNNException("in_unitsが1未満");
			if (params.hiddenUnits < 1)
				throw XNNException("hidden_unitsが1未満");
			if (params.outUnits < 1)
				throw XNNException("out_unitsが1未満");
			if (params.hiddenLayers < 1)
				throw XNNException("hidden_layersが1未満");

			layers.clear();

			if (params.scaleInput != 0)
				layers.emplace_back(new InputScalingLayer(params.inUnits));

			layers.emplace_back(
				new FullyConnectedLayer<ActivationFunction::ReLU>(
					params.inUnits, params.hiddenUnits));

			for (size_t i = 0; i < params.hiddenLayers - 1; i++)
				layers.emplace_back(
					new FullyConnectedLayer<ActivationFunction::ReLU>(
						params.hiddenUnits, params.hiddenUnits));

			if (params.objective == XNNObjective::MultiSoftmax)
				layers.emplace_back(
					new FullyConnectedLayer<ActivationFunction::Identity>(
						params.hiddenUnits, params.outUnits));
			else
				layers.emplace_back(
					new FullyConnectedLayer<ActivationFunction::Sigmoid>(
						params.hiddenUnits, params.outUnits));

			if (params.objective == XNNObjective::MultiSoftmax)
				layers.emplace_back(new SoftmaxLayer());
		}
		// データのチェック・変換
		void CheckAndTransform(vector<XNNData>& data) const {
			for (auto& d : data) {
				if (d.in.size() != params.inUnits)
					throw XNNException("入力ユニット数と入力データの次元数が不一致: " +
						to_string(params.inUnits) + " => " +
						to_string(d.in.size()));
				if (params.objective == XNNObjective::MultiSoftmax) {
					if (d.out.size() != 1)
						throw XNNException("他クラス分類でラベルが1つでない: ラベル数=" +
							to_string(d.out.size()));
					// クラスのindexからone hot vector形式に変換
					int classIndex = (int)round(d.out[0]);
					if (classIndex < 0 || params.outUnits <= classIndex)
						throw XNNException("ラベルが不正: ラベル=" +
							to_string(classIndex) + " 範囲=[0, " +
							to_string(params.outUnits) + ")");
					d.out[0] = 0;
					d.out.resize(params.outUnits, 0);
					d.out[classIndex] = 1;
				} else {
					if (d.out.size() != params.outUnits)
						throw XNNException("出力ユニット数とラベルの数が不一致: " +
							to_string(params.outUnits) + " => " +
							to_string(d.out.size()));
				}
			}
		}
		// 学習
		void Train(vector<XNNData>&& trainData, vector<XNNData>&& testData) {
			mt.seed(5489); // fixed seed

			CheckAndTransform(trainData);
			CheckAndTransform(testData);
			if (1 <= params.verbose) {
				cout << "訓練データ: " << trainData.size() << "件" << endl;
				cout << "検証データ: " << testData.size() << "件" << endl;
			}

			// シャッフル
			shuffle(trainData.begin(), trainData.end(), mt);
			// 重みなどの初期化
			{
				double inputNorm = NAN, outputNorm = NAN; // 初期値は使わないので適当
				if (params.scaleInput == 0) { // スケーリングしない場合初期値が必要なので頑張って算出する。
					inputNorm = 0.0;
					for (size_t c = 0; c < trainData.size(); c++)
						for (size_t i = 0; i < trainData[c].in.size(); i++)
							inputNorm += abs(trainData[c].in[i]);
					inputNorm /= trainData.size();
				}
				for (auto& l : layers) {
					l->Initialize(trainData, mt, inputNorm, outputNorm);
					swap(inputNorm, outputNorm);
				}
			}
			// 学習の準備
			trainers.clear();
			for (auto& l : layers)
				trainers.push_back(l->CreateTrainer());

			// データが足りなければ増やす (整数倍に繰り返す)
			// 最低10万を1セットとする。
			size_t originalSize = trainData.size();
			while (trainData.size() < params.minMiniBatchCount * params.miniBatchSize) {
				for (size_t i = 0; i < originalSize; i++)
					trainData.push_back(trainData[i]);
			}

			// 設定の確認のためネットワークの大きさを表示
			cout << "ネットワーク: " << params.inUnits
				<< " - (" << params.hiddenUnits << " x " << params.hiddenLayers
				<< ") - " << params.outUnits
				<< endl;
			if (1 <= params.verbose)
				cout << "学習開始" << endl;

			// 学習を回す。検証誤差がほぼ下がらなくなったら終了。

			// ＜程よく検証するために色々考えてみた謎条件＞
			// テストデータが1万以下の場合、訓練データ10万毎に検証を入れる。
			// ただし訓練データが20万未満の場合、訓練データ1周毎にする。
			// テストデータが1万以上なら、比例する感じで同様に。
			auto testSize = max(testData.size(), (size_t)10000);
			auto testIntervalEpoch = testSize * 10 / params.miniBatchSize;
			auto testOnlyEnd = trainData.size() < (testIntervalEpoch * 2 * params.miniBatchSize);

			vector<double> lastRMSE(params.outUnits, numeric_limits<double>::max());
			auto toBeStop = [&](bool checkStop) {
				// 訓練誤差・検証誤差の算出
				auto pred1 = Predict(trainData, 0, min(trainData.size(), testSize));
				auto pred2 = Predict(testData, 0, testData.size());
				assert(lastRMSE.size() == pred2.size());
				// 表示
				for (size_t o = 0; o < pred2.size(); o++)
					cout << "検証: out[" << o << "] :"
					<< " train={" << pred1[o].ToString() << "}"
					<< " test={" << pred2[o].ToString() << "}"
					<< endl;
				// 終了判定
				if (checkStop) {
					// 検証データのRMSEの差がstopDeltaRMSE未満になったら学習終了。
					// (ただし最低でも訓練データを1回は使用する。)
					bool stopAll = true;
					for (size_t o = 0; o < pred2.size(); o++) {
						auto rmse = pred2[o].GetRMSE();
						auto delta = lastRMSE[o] - rmse;
						lastRMSE[o] = rmse;
						if (params.stopDeltaRMSE <= delta &&
							params.stopDeltaRMSE <= rmse) // 0 <= rmseなので充分小さければ止まっていい
							stopAll = false;
					}
					return stopAll;
				}
				return false;
			};

			for (size_t loop = 0; ; loop++) {
				// シャッフル
				shuffle(trainData.begin(), trainData.end(), mt);

				// 学習
				{
					ProgressTimer timer;

					const size_t MaxEpoch = trainData.size() / params.miniBatchSize;
					for (size_t epoch = 0; epoch < MaxEpoch; epoch++) {
						PartialFit(trainData, epoch * params.miniBatchSize, params.miniBatchSize);

						if (1 <= params.verbose && (epoch + 1) % (MaxEpoch / 10) == 0) {
							timer.Set(epoch + 1, MaxEpoch);
							cout << "学習:"
								<< " loop=" << loop
								<< " " << timer.ToStringCount()
								<< " " << timer.ToStringTime()
								<< endl;
						}
						// ある程度大きい場合は途中で検証
						if (!testOnlyEnd && (epoch + 1) % testIntervalEpoch == 0) {
							if (toBeStop(0 < loop)) // 終了は最低でも一周した後。
								break;
						}
					}
				}
				// 小さい場合は1周毎に検証
				if (testOnlyEnd)
					if (toBeStop(true))
						break;
			}

			if (1 <= params.verbose)
				cout << "学習終了" << endl;
		}
		// ミニバッチによる更新
		void PartialFit(const std::vector<XNNData>& trainData, size_t startIndex, size_t count) {
#pragma omp parallel for
			for (int i = 0; i < (int)trainers.size(); i++)
				trainers[i]->Clear();

			atomic<int> miniBatchIndex(0);
#pragma omp parallel
			{
				vector<vector<float>> out(layers.size() + 1);
				vector<float> errorIn, errorOut;
				for (auto& o : out)
					o.reserve(params.hiddenUnits);
				errorIn.reserve(params.hiddenUnits);
				errorOut.reserve(params.hiddenUnits);

				while (true) {
					int index = miniBatchIndex++;
					if (count <= index)
						break;
					auto& data = trainData[startIndex + index];

					out[0] = data.in;
					// 順伝搬
					for (size_t i = 0; i < layers.size(); i++)
						layers[i]->Forward(out[i], out[i + 1]);
					// エラーを算出
					// ロジスティック回帰／線形二乗誤差：教師 - 予測
					errorIn = data.out;
					for (size_t i = 0; i < errorIn.size(); i++)
						errorIn[i] = errorIn[i] - out.back()[i];
					// 逆伝搬
					for (int i = (int)layers.size() - 1; 0 <= i; i--) {
						layers[i]->Backward(*trainers[i],
							out[i], out[i + 1],
							errorOut, errorIn);
						swap(errorIn, errorOut);
					}
				}
			}

#pragma omp parallel for
			for (int i = 0; i < (int)trainers.size(); i++)
				trainers[i]->Update();
		}

		// 予測
		vector<RegressionScore> Predict(const std::vector<XNNData>& testData, size_t startIndex, size_t count) const {
			vector<RegressionScore> score(params.outUnits);
#pragma omp parallel for
			for (int i = 0; i < (int)count; i++) {
				auto& data = testData[startIndex + i];
				auto pred = Predict(vector<float>(data.in));
				assert(pred.size() == params.outUnits);
				assert(data.out.size() == params.outUnits);
				for (size_t o = 0; o < pred.size(); o++)
					score[o].Add(data.out[o] - pred[o]);
			}
			return score;
		}

		// 予測
		XNNModel::PredictResult Predict(vector<XNNData>&& testData) {
			CheckAndTransform(testData);

			vector<unique_ptr<IScore>> score;
			switch (params.objective) {
			case XNNObjective::RegLogistic:
				for (size_t i = 0; i < params.outUnits; i++)
					score.emplace_back(new RegressionScore());
				break;
			case XNNObjective::BinaryLogistic:
				for (size_t i = 0; i < params.outUnits; i++)
					score.emplace_back(new ClassificationScore(2));
				break;
			case XNNObjective::MultiSoftmax:
				score.emplace_back(new ClassificationScore(params.outUnits));
				break;
			default:
				assert(false);
			}

			stringstream stat, raw;
			raw << fixed << setprecision(7);
			for (int i = 0; i < (int)testData.size(); i++) {
				auto& data = testData[i];
				auto pred = Predict(vector<float>(data.in));
				assert(pred.size() == params.outUnits);
				assert(data.out.size() == params.outUnits);
				if (params.objective == XNNObjective::MultiSoftmax)
					score[0]->Add(data.out, pred);
				else
					for (size_t o = 0; o < pred.size(); o++)
						score[o]->Add(data.out[o], pred[o]);
				for (size_t o = 0; o < pred.size(); o++) {
					if (0 < o)
						raw << " ";
					raw << pred[o];
				}
				raw << endl;
			}

			for (size_t o = 0; o < score.size(); o++) {
				if (2 <= score.size())
					stat << "--------------------------- out[" << o << "] ---------------------------" << endl;
				stat << score[o]->ToString();
				if (params.objective == XNNObjective::RegLogistic)
					stat << endl; // TODO:そのうちなんとかする
			}

			return{ stat.str(), raw.str() };
		}

		// 予測
		vector<float> Predict(vector<float>&& in) const {
			vector<float> out;
			out.reserve(params.hiddenUnits);
			for (size_t i = 0; i < layers.size(); i++) {
				layers[i]->Forward(in, out);
				swap(in, out);
			}
			return in;
		}
	};

	vector<XNNData> LoadSVMLight(const string& path, int inUnits) {
		ifstream is(path);
		if (!is)
			throw XNNException(path + "が開けませんでした。");
		return LoadSVMLight(is, inUnits);
	}
	vector<XNNData> LoadSVMLight(istream& is, int inUnits) {
		vector<XNNData> result;
		for (string line; getline(is, line); ) {
			XNNData data;
			stringstream ss(line);
			string token;
			// ラベル
			ss >> skipws >> token;
			replace(token.begin(), token.end(), ',', ' ');
			stringstream ssLabel(token);
			for (float label; ssLabel >> label;)
				data.out.push_back(label);
			// 特徴
			data.in.resize(inUnits, 0.0f);
			while (ss >> token) {
				auto coron = token.find(':');
				if (coron != string::npos) {
					auto index = stoll(token.substr(0, coron));
					if (index <= 0)
						throw XNNException("特徴のインデックスが0以下: 行の内容=" + line);
					if (inUnits < index)
						throw XNNException("特徴のインデックスが" + to_string(inUnits + 1) + "以上: 行の内容=" + line);
					data.in[index - 1] = stof(token.substr(coron + 1));
				}
			}
			result.emplace_back(move(data));
		}
		return result;
	}
	void SaveSVMLight(const string& path, const vector<XNNData>& data) {
		ofstream os(path);
		if (!os)
			throw XNNException(path + "が開けませんでした。");
		SaveSVMLight(os, data);
	}
	void SaveSVMLight(ostream& os, const vector<XNNData>& data) {
		for (auto& d : data) {
			for (size_t i = 0; i < d.out.size(); i++) {
				if (0 < i)
					os << ",";
				os << d.out[i];
			}
			for (size_t i = 0; i < d.in.size(); i++)
				os << " " << (i + 1) << ":" << d.in[i];
			os << endl;
		}
	}

	XNNModel::XNNModel(const XNNParams& params)
		: impl(new XNNImpl(params)) { }

	XNNModel::XNNModel(const string& path)
		: impl(new XNNImpl(path)) { }

	XNNModel::~XNNModel() {}

	void XNNModel::Save(const string& path) const {
		impl->Save(path);
	}
	void XNNModel::Load(const string& path) {
		impl->Load(path);
	}
	void XNNModel::Train(vector<XNNData>&& trainData, vector<XNNData>&& testData) {
		impl->Train(move(trainData), move(testData));
	}
	XNNModel::PredictResult XNNModel::Predict(vector<XNNData>&& testData) const {
		return impl->Predict(move(testData));
	}
	vector<float> XNNModel::Predict(vector<float>&& in) const {
		return impl->Predict(move(in));
	}
}
