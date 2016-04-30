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
#include <cfloat>

#ifdef _MSC_VER
#pragma warning(disable : 4127) // 条件式が定数です。
#pragma warning(disable : 4100) // 引数は関数の本体部で 1 度も参照されません。
#endif

namespace XNN {
	using namespace std::chrono;

	// 雑なoperator色々
	template<class Range, class T>
	void operator*=(Range& x, T y) { for (auto& r : x) r *= y; }
	template<class Range, class T>
	void operator/=(Range& x, T y) { for (auto& r : x) r /= y; }
	template<class T>
	void operator+=(vector<T>& x, const vector<T>& y) {
		assert(x.size() == y.size());
		for (size_t i = 0, n = x.size(); i < n; i++)
			x[i] += y[i];
	}
	template<class T>
	void operator-=(vector<T>& x, const vector<T>& y) {
		assert(x.size() == y.size());
		for (size_t i = 0, n = x.size(); i < n; i++)
			x[i] -= y[i];
	}
	template<class T>
	void operator*=(vector<T>& x, const vector<T>& y) {
		assert(x.size() == y.size());
		for (size_t i = 0, n = x.size(); i < n; i++)
			x[i] *= y[i];
	}
	template<class T>
	void operator/=(vector<T>& x, const vector<T>& y) {
		assert(x.size() == y.size());
		for (size_t i = 0, n = x.size(); i < n; i++)
			x[i] /= y[i];
	}

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
			Add(size_t(0.5 <= label), size_t(0.5 <= pred));
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
			for (size_t classIndex = 0; classIndex < list.size(); classIndex++)
				total += GetLabelCount(classIndex);
			int indexWidth = (int)floor(log10(list.size() - 1)) + 1;
			stringstream ss;
			ss << fixed << setprecision(1);
			double avgPrec = 0, avgRecl = 0, avgFval = 0;
			for (size_t classIndex = 0; classIndex < list.size(); classIndex++) {
				auto prec = GetPrecision(classIndex);
				auto recl = GetRecall(classIndex);
				auto fval = GetFValue(classIndex);
				auto pick = (double)GetPickCount(classIndex) / total;
				auto dist = (double)GetLabelCount(classIndex) / total;
				ss << "class[" << setw(indexWidth) << classIndex << "]:"
					<< " 適合率=" << setw(4) << prec * 100 << "%"
					<< " 再現率=" << setw(4) << recl * 100 << "%"
					<< " F値=" << setw(4) << fval * 100 << "%"
					<< " 選択率=" << setw(4) << pick * 100 << "%"
					<< " 分布=" << setw(4) << dist * 100 << "%"
					<< endl;
				avgPrec += prec * dist;
				avgRecl += recl * dist;
				avgFval += fval * dist;
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
		double l1 = 0, l2 = 0;
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

		AdamOptimizer(size_t dimension, double scale) : OptimizerBase(dimension, scale), m(new double[dimension]), v(new double[dimension]) {
			fill(m.get(), m.get() + dimension, 0.0);
			fill(v.get(), v.get() + dimension, 0.0);
		}

		double GetStep(size_t i, double g, uint64_t lt) {
			const double alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
			m[i] = beta1 * m[i] + (1.0 - beta1) * g;
			v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
			auto hm = m[i] / (1.0 - pow(beta1, lt));
			auto hv = v[i] / (1.0 - pow(beta2, lt));
			return alpha * hm / (sqrt(hv) + epsilon);
		}
	};

	// 層を学習するクラス
	struct ILayerTrainer {
		virtual ~ILayerTrainer() {}
		// 勾配をクリアする
		virtual void Clear() = 0;
		// 勾配に従って重みを更新
		virtual void Update() = 0;
	};
	// 学習するものが無い層用のILayerTrainer
	struct NullLayerTrainer : public ILayerTrainer {
		void Clear() override {}
		void Update() override {}
	};

	// 層
	struct ILayer {
		virtual ~ILayer() {}
		// モデルの読み込み
		virtual void Load(istream& s) {}
		// モデルの保存
		virtual void Save(ostream& s) const {}
		// 学習用に初期化
		// 学習率を程よくするために、入力のL1ノルムの平均(の推定値)を受け取り、出力のL1ノルムの平均(の推定値)を返す。(順に伝搬させる)
		virtual void Initialize(const vector<XNNData>& data, mt19937_64& rnd, double inputNorm, double& outputNorm) = 0;
		// 学習するクラスを作る
		virtual unique_ptr<ILayerTrainer> CreateTrainer(const XNNParams& params, mt19937_64& rnd) {
			return unique_ptr<ILayerTrainer>(new NullLayerTrainer());
		}
		// 順伝搬
		virtual void Forward(const vector<float>& in, vector<float>& out, ILayerTrainer* trainer) const {
			out = in;
		}
		// 逆伝搬
		virtual void Backward(
			ILayerTrainer& trainer,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const {
			errorOut = errorIn;
		}
	};

	// 入力をスケーリングする層
	struct InputScalingLayer : public ILayer {
		uint64_t inUnits;
		vector<float> scale;
		InputScalingLayer(uint64_t inUnits, float minScale = 1.0f) : inUnits(inUnits), scale(inUnits, minScale) {}
		// モデルの読み込み
		void Load(istream& s) override {
			s.read((char*)&inUnits, sizeof inUnits);
			scale.resize(inUnits);
			s.read((char*)&scale[0], scale.size() * sizeof scale[0]);
		}
		// モデルの保存
		void Save(ostream& s) const override {
			s.write((const char*)&inUnits, sizeof inUnits);
			s.write((const char*)&scale[0], scale.size() * sizeof scale[0]);
		}
		// 学習用に初期化
		// 学習率を程よくするために、入力のL1ノルムの平均(の推定値)を受け取り、出力のL1ノルムの平均(の推定値)を返す。(順に伝搬させる)
		void Initialize(const vector<XNNData>& data, mt19937_64& rnd, double inputNorm, double& outputNorm) override {
			// スケールの決定。とりあえず絶対値の最大値で割り算する感じにする。
			for (auto& d : data) {
				for (size_t i = 0; i < inUnits; i++) {
					auto a = abs(d.in[i]);
					if (scale[i] < a)
						scale[i] = a;
				}
			}
			for (size_t i = 0; i < inUnits; i++)
				scale[i] = 1.0f / scale[i];
			// 出力のL1ノルム
			double l1 = 0;
			for (auto& d : data) {
				for (size_t i = 0; i < inUnits; i++) {
					auto x = d.in[i] * scale[i];
					if (isnormal(x))
						l1 += abs(x);
				}
			}
			outputNorm = l1 / data.size();
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out, ILayerTrainer* trainer) const override {
			out = in;
			out *= scale;
		}
	};

	// 全層結合層
	struct FullyConnectedLayer : public ILayer {
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
			inputNorm = inputNorm_;
			auto s = sqrt(3.0f / (float(inputNorm * 2)));
			auto nd = uniform_real_distribution<float>(-s, s);
			for (auto& x : weights)
				x = nd(rnd);
			for (auto& x : biases)
				x = 0.1f / inUnits; // バイアスは適当な正の定数。dead neurons対策。
			outputNorm = (double)outUnits;
		}
		// 学習するクラス
		struct Trainer : ILayerTrainer {
			FullyConnectedLayer& owner;
			AdamOptimizer optimizerW, optimizerB;
			DenseVector<> gradW, gradB;
			Trainer(FullyConnectedLayer& owner, const XNNParams& params) :
				owner(owner),
				// 入力の大きさに応じて謎チューニング
				optimizerW(owner.weights.size(), 10.0 / sqrt(owner.inputNorm)),
				optimizerB(owner.biases.size(), 10.0 / sqrt(owner.inputNorm)),
				gradW(owner.weights.size()),
				gradB(owner.biases.size()) {
				// 正則化項
				optimizerW.l1 = params.l1;
				optimizerW.l2 = params.l2;
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
		unique_ptr<ILayerTrainer> CreateTrainer(const XNNParams& params, mt19937_64& rnd) override {
			return unique_ptr<ILayerTrainer>(new Trainer(*this, params));
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out, ILayerTrainer* trainer) const override {
			assert(in.size() == inUnits);
			out.clear();
			out.resize(outUnits, 0.0f);
			for (size_t o = 0; o < outUnits; o++) {
				for (size_t i = 0; i < inUnits; i++)
					out[o] += in[i] * weights[o * inUnits + i];
			}
			out += biases;
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
		// 特徴の重要度のようなもの
		vector<float> GetWeightRMS() const {
			vector<float> rms(inUnits, 0.0f);
			for (size_t o = 0; o < outUnits; o++)
				for (size_t i = 0; i < inUnits; i++)
					rms[i] += weights[o * inUnits + i] * weights[o * inUnits + i];
			for (auto& x : rms)
				x = sqrt(x / outUnits);
			return rms;
		}
	};

	// 活性化関数レイヤー
	template<XNNActivation Act>
	struct ActivationLayer : public ILayer {
		ActivationLayer(uint64_t inUnits) {}
		// 学習用に初期化
		void Initialize(const vector<XNNData>& data, mt19937_64& rnd, double inputNorm, double& outputNorm) override {
			switch (Act) {
			case XNNActivation::ReLU: outputNorm = inputNorm / 2; break;
			case XNNActivation::Sigmoid: outputNorm = inputNorm / 2; break;
			case XNNActivation::Softmax: outputNorm = 1.0; break;
			}
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out, ILayerTrainer* trainer) const override {
			out = in;
			switch (Act) {
			case XNNActivation::ReLU:
				for (auto& x : out)
					x = max(0.f, x);
				break;
			case XNNActivation::Sigmoid:
				for (auto& x : out)
					x = 1.0f / (1.0f + exp(-x));
				break;
			case XNNActivation::Identity:
				break;
			case XNNActivation::Softmax:
				{
					auto m = *max_element(out.begin(), out.end());
					for (auto& x : out)
						x = exp(x - m); // 最大値を引く(オーバーフロー対策)
					auto s = accumulate(out.begin(), out.end(), 0.0f);
					out /= s;
				}
				break;
			}
		}
		// 逆伝搬
		void Backward(
			ILayerTrainer& trainer,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const override {
			errorOut = errorIn;
			assert(errorOut.size() == out.size());
			switch (Act) {
			case XNNActivation::ReLU:
				for (size_t o = 0; o < errorOut.size(); o++)
					if (in[o] <= 0)
						errorOut[o] = 0.0f;
				break;
			case XNNActivation::Sigmoid:
			case XNNActivation::Identity:
			case XNNActivation::Softmax:
				// 中間層の場合はここで微分する必要があるが、
				// 現状は出力層限定なのでそのまま伝搬させる。
				break;
			}
		}
	};
	template<>
	struct ActivationLayer<XNNActivation::PReLU> : public ILayer {
		const uint64_t inUnits;
		vector<float> weights;
		ActivationLayer(uint64_t inUnits) : inUnits(inUnits), weights(inUnits, 0.25f) {}
		// モデルの読み込み
		void Load(istream& s) override {
			s.read((char*)&inUnits, sizeof inUnits);
			weights.resize(inUnits);
			s.read((char*)&weights[0], weights.size() * sizeof weights[0]);
		}
		// モデルの保存
		void Save(ostream& s) const override {
			s.write((const char*)&inUnits, sizeof inUnits);
			s.write((const char*)&weights[0], weights.size() * sizeof weights[0]);
		}
		// 学習用に初期化
		void Initialize(const vector<XNNData>&, mt19937_64&, double inputNorm, double& outputNorm) override {
			for (auto& x : weights)
				x = 0.25f;
			outputNorm = inputNorm / 2;
		}
		// 学習するクラス
		struct Trainer : ILayerTrainer {
			ActivationLayer<XNNActivation::PReLU>& owner;
			AdamOptimizer optimizerW;
			DenseVector<> gradW;
			Trainer(ActivationLayer<XNNActivation::PReLU>& owner) :
				owner(owner),
				optimizerW(owner.weights.size(), 1),
				gradW(owner.weights.size()) {
			}
			void Clear() override {
				gradW.Clear();
			}
			void Update() override {
				optimizerW.Update(owner.weights, gradW);
			}
		};
		// 学習するクラスを作る
		unique_ptr<ILayerTrainer> CreateTrainer(const XNNParams& params, mt19937_64& rnd) override {
			return unique_ptr<ILayerTrainer>(new Trainer(*this));
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out, ILayerTrainer* trainer) const override {
			assert(in.size() == inUnits);
			out = in;
			for (size_t i = 0; i < inUnits; i++)
				if (in[i] < 0)
					out[i] *= weights[i];
		}
		// 逆伝搬
		void Backward(ILayerTrainer& trainer_,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const override {
			assert(in.size() == inUnits);
			assert(out.size() == inUnits);
			assert(errorIn.size() == inUnits);
			errorOut = errorIn;
			auto& trainer = (Trainer&)trainer_;
			for (size_t i = 0; i < inUnits; i++) {
				if (in[i] < 0) {
					// 誤差の伝播
					errorOut[i] *= weights[i];
					// wの勾配
					trainer.gradW += make_pair(i, errorIn[i] * in[i]);
				}
			}
		}
	};
	struct DropoutLayer : public ILayer {
		const uint64_t inUnits;
		const float keepProb;
		DropoutLayer(uint64_t inUnits, float keepProb) : inUnits(inUnits), keepProb(keepProb) {}
		// 学習用に初期化
		void Initialize(const vector<XNNData>&, mt19937_64&, double inputNorm, double& outputNorm) override {
			outputNorm = inputNorm / 2;
		}
		struct Trainer : public ILayerTrainer {
			vector<mt19937_64> rnds;
			vector<unique_ptr<bool[]>> keepFlags;
			Trainer(mt19937_64& rnd, uint64_t inUnits) {
				auto threadCount = omp_get_max_threads();
				array<mt19937_64::result_type, 16> seed;
				for (int i = 0; i < threadCount; i++) {
					generate(seed.begin(), seed.end(), ref(rnd));
					seed_seq seq(seed.begin(), seed.end());
					rnds.emplace_back(mt19937_64(seq));
					keepFlags.emplace_back(new bool[inUnits]);
				}
			}
			void Clear() override {}
			void Update() override {}
		};
		// 学習するクラスを作る
		unique_ptr<ILayerTrainer> CreateTrainer(const XNNParams& params, mt19937_64& rnd) {
			return unique_ptr<ILayerTrainer>(new Trainer(rnd, inUnits));
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out, ILayerTrainer* trainer_) const override {
			assert(in.size() == inUnits);
			out = in;
			if (trainer_ != nullptr) {
				// ランダムにdropする
				auto tid = omp_get_thread_num();
				auto& trainer = (Trainer&)*trainer_;
				auto rnd = trainer.rnds[tid];
				auto keepFlags = trainer.keepFlags[tid].get();

				auto keepCount = (size_t)round(inUnits * keepProb);
				for (size_t i = 0; i < keepCount; i++)
					keepFlags[i] = true;
				for (size_t i = keepCount; i < inUnits; i++)
					keepFlags[i] = false;
				shuffle(keepFlags, keepFlags + inUnits, rnd);

				for (size_t i = 0; i < inUnits; i++)
					if (keepFlags[i])
						out[i] /= keepProb; // keep
					else
						out[i] = 0; // drop
			}
		}
		// 逆伝搬
		void Backward(ILayerTrainer& trainer_,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const override {
			assert(in.size() == inUnits);
			assert(out.size() == inUnits);
			assert(errorIn.size() == inUnits);
			auto tid = omp_get_thread_num();
			auto& trainer = (Trainer&)trainer_;
			auto keepFlags = trainer.keepFlags[tid].get();
			errorOut = errorIn;
			for (size_t i = 0; i < inUnits; i++) {
				if (keepFlags[i])
					errorOut[i] /= keepProb; // keep
				else
					errorOut[i] = 0; // drop
			}
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
			Save(fs);
		}
		void Save(ostream& fs) const {
			fs.write((const char*)&params, sizeof params);
			for (auto& l : layers)
				l->Save(fs);
		}
		void Load(const string& path) {
			ifstream fs(path, ios_base::binary);
			if (!fs)
				throw XNNException(path + "が開けませんでした。");
			Load(fs);
		}
		void Load(istream& fs) {
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
			if (params.hiddenLayers < 0)
				throw XNNException("hidden_layersが0未満");

			layers.clear();

			// 前処理の層
			if (params.scaleInput != 0)
				layers.emplace_back(new InputScalingLayer(params.inUnits));

			// 入力層・中間層
			for (int i = 0; i < params.hiddenLayers; i++) {
				int inUnits = i == 0 ? params.inUnits : params.hiddenUnits;
				int outUnits = params.hiddenUnits;
				layers.emplace_back(new FullyConnectedLayer(inUnits, outUnits));
				if (params.activation == XNNActivation::ReLU)
					layers.emplace_back(new ActivationLayer<XNNActivation::ReLU>(outUnits));
				else if (params.activation == XNNActivation::PReLU)
					layers.emplace_back(new ActivationLayer<XNNActivation::PReLU>(outUnits));
				else
					throw XNNException("activationが不正: " + ToString(params.activation));
				if (params.dropoutKeepProb < 1.0f) {
					if (params.dropoutKeepProb < 0.0f)
						throw XNNException("dropout_keep_probが不正: " + to_string(params.dropoutKeepProb));
					layers.emplace_back(new DropoutLayer(outUnits, params.dropoutKeepProb));
				}
			}
			// 出力層
			{
				int inUnits = params.hiddenLayers <= 0 ? params.inUnits : params.hiddenUnits;
				int outUnits = params.outUnits;
				layers.emplace_back(new FullyConnectedLayer(inUnits, outUnits));
				if (params.objective == XNNObjective::MultiSoftmax)
					layers.emplace_back(new ActivationLayer<XNNActivation::Softmax>(outUnits));
				else if (params.objective == XNNObjective::RegLinear)
					layers.emplace_back(new ActivationLayer<XNNActivation::Identity>(outUnits));
				else
					layers.emplace_back(new ActivationLayer<XNNActivation::Sigmoid>(outUnits));
			}
		}
		// データのチェック
		void CheckData(vector<XNNData>& data) const {
			for (auto& d : data) {
				if ((int)d.in.size() != params.inUnits)
					throw XNNException("入力ユニット数と入力データの次元数が不一致: " +
						to_string(params.inUnits) + " => " +
						to_string(d.in.size()));
				if (params.objective == XNNObjective::MultiSoftmax) {
					if (d.out.size() != 1)
						throw XNNException("他クラス分類でラベルが1つでない: ラベル数=" +
							to_string(d.out.size()));
				} else {
					if ((int)d.out.size() != params.outUnits)
						throw XNNException("出力ユニット数とラベルの数が不一致: " +
							to_string(params.outUnits) + " => " +
							to_string(d.out.size()));
				}
			}
		}
		// 多クラス分類の場合に、クラスのindexからone hot vector形式に変換
		void TransformOutput(vector<float>& out) const {
			if (params.objective == XNNObjective::MultiSoftmax) {
				// クラスのindexからone hot vector形式に変換
				int classIndex = (int)round(out[0]);
				if (classIndex < 0 || params.outUnits <= classIndex)
					throw XNNException("ラベルが不正: ラベル=" +
						to_string(classIndex) + " 範囲=[0, " +
						to_string(params.outUnits) + ")");
				out[0] = 0;
				out.resize(params.outUnits, 0);
				out[classIndex] = 1;
			}
			assert((int)out.size() == params.outUnits);
		}
		// 学習
		void Train(vector<XNNData>&& trainData, vector<XNNData>&& testData) {
			auto startTime = high_resolution_clock::now();
			mt.seed(5489); // fixed seed

			CheckData(trainData);
			CheckData(testData);
			if (1 <= params.verbose) {
				cout << "訓練データ: " << trainData.size() << "件" << endl;
				cout << "検証データ: " << testData.size() << "件" << endl;
			}

			if (params.objective == XNNObjective::BinaryLogistic) {
				// scale_pos_weightの自動設定
				if (params.scalePosWeight < 0) {
					if (params.outUnits == 1) {
						array<size_t, 2> counts = { { 0, 0 } };
						for (auto& d : trainData)
							counts[(int)round(d.out[0])]++;
						params.scalePosWeight = (float)counts[0] / counts[1];
					} else {
						params.scalePosWeight = 1;
					}
				}
				cout << "scale_pos_weight = " << params.scalePosWeight << endl;
			}

			// シャッフル
			shuffle(trainData.begin(), trainData.end(), mt);
			// 重みなどの初期化
			{
				double inputNorm = NAN, outputNorm = NAN; // 初期値は使わないので適当
				if (params.scaleInput == 0) { // スケーリングしない場合初期値が必要なので頑張って算出する。
					inputNorm = 0.0;
					for (auto& d : trainData)
						for (size_t i = 0; i < d.in.size(); i++)
							if (isnormal(d.in[i]))
								inputNorm += abs(d.in[i]);
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
				trainers.push_back(l->CreateTrainer(params, mt));

			// データが足りなければ増やす (整数倍に繰り返す)
			// 最低10万を1セットとする。
			size_t originalSize = trainData.size();
			while (trainData.size() < (size_t)params.minMiniBatchCount * params.miniBatchSize) {
				for (size_t i = 0; i < originalSize; i++)
					trainData.push_back(trainData[i]);
			}

			// 設定の確認のためネットワークの大きさを表示
			cout << "ネットワーク: " << params.inUnits
				<< ":" << ToString(params.activation)
				<< " - (" << params.hiddenUnits
				<< ":" << ToString(params.activation)
				<< " x " << params.hiddenLayers
				<< ") - " << params.outUnits
				<< endl;
			if (1 <= params.verbose)
				cout << "学習開始" << endl;

			// 学習を回す。検証誤差がほぼ下がらなくなったら終了。

			double minRMSE = numeric_limits<double>::max();
			int earlyStoppingCount = 0;
			stringstream bestModel;

			size_t testSize = min(max(testData.size(), (size_t)10000), trainData.size());

			for (size_t epoch = 1; ; epoch++) {
				// 学習
				{
					ProgressTimer timer;

					const size_t MaxMB = trainData.size() / params.miniBatchSize;
					RegressionScore score;
					for (size_t mb = 0; mb < MaxMB; mb++) {
						score += PartialFit(trainData, mb * params.miniBatchSize, params.miniBatchSize);

						if (1 <= params.verbose && (mb + 1) % (MaxMB / 10) == 0) {
							timer.Set(mb + 1, MaxMB);
							cout << "学習:"
								<< " epoch=" << epoch
								<< " " << timer.ToStringCount()
								<< " train={" << score.ToString() << "}"
								<< " " << timer.ToStringTime()
								<< endl;
							score.Clear();
						}
					}
				}

				// 訓練誤差・検証誤差の算出
				shuffle(trainData.begin(), trainData.begin() + testSize, mt);
				auto pred1 = Predict(trainData, 0, testSize);
				auto pred2 = Predict(testData, 0, testData.size());
				assert(pred1.size() == pred2.size());

				// 表示
				if (1 <= params.verbose || pred2.size() <= 1) {
					for (size_t o = 0; o < pred2.size(); o++)
						cout << "検証: epoch=" << epoch
						<< " out[" << o << "] :"
						<< " train={" << pred1[o].ToString() << "}"
						<< " test={" << pred2[o].ToString() << "}"
						<< endl;
				}
				RegressionScore average1, average2;
				for (size_t o = 0; o < pred2.size(); o++) {
					average1 += pred1[o];
					average2 += pred2[o];
				}
				if (1 < pred2.size()) {
					cout << "検証: epoch=" << epoch
						<< " average:"
						<< " train={" << average1.ToString() << "}"
						<< " test={" << average2.ToString() << "}"
						<< endl;
				}

				// 終了判定
				auto rmse = average2.GetRMSE();
				if (rmse < 0.005)
					break; // 充分小さければ止まる
				if (rmse < minRMSE) {
					// 最高記録を更新
					minRMSE = rmse;
					earlyStoppingCount = 0;
					// モデルを保存
					bestModel.str(""); // 空にする
					Save(bestModel);
				} else {
					if (params.earlyStoppingTolerance < ++earlyStoppingCount) {
						// 最高記録のモデルを復元
						Load(bestModel);
						break;
					}
				}

				// シャッフル
				shuffle(trainData.begin(), trainData.end(), mt);
			}

			auto dt = high_resolution_clock::now() - startTime;
			cout << "学習完了: " << (duration_cast<milliseconds>(dt).count() / 1000.0) << "秒" << endl;
		}
		// ミニバッチによる更新
		RegressionScore PartialFit(const std::vector<XNNData>& trainData, size_t startIndex, size_t count) {
#pragma omp parallel for
			for (int i = 0; i < (int)trainers.size(); i++)
				trainers[i]->Clear();

			// scale_pos_weightの比率を保ちつつ、あまり極端な値にならないスケールを算出
			// 例: scale_pos_weight=1なら{ 1, 1 }、10なら{ 0.1818, 1.818 }。
			const array<float, 2> binaryScales = { {
				2 / (params.scalePosWeight + 1),
				params.scalePosWeight * 2 / (params.scalePosWeight + 1),
			} };

			RegressionScore score;
			atomic<size_t> miniBatchIndex(0);
#pragma omp parallel
			{
				vector<vector<float>> out(layers.size() + 1);
				vector<float> errorIn, errorOut;
				for (auto& o : out)
					o.reserve(params.hiddenUnits);
				errorIn.reserve(params.hiddenUnits);
				errorOut.reserve(params.hiddenUnits);

				while (true) {
					auto index = miniBatchIndex++;
					if (count <= index)
						break;
					auto& data = trainData[startIndex + index];

					out[0] = data.in;
					// 順伝搬
					for (size_t i = 0; i < layers.size(); i++)
						layers[i]->Forward(out[i], out[i + 1], trainers[i].get());
					// エラーを算出
					// ロジスティック回帰／線形二乗誤差：教師 - 予測
					errorIn = data.out;
					TransformOutput(errorIn);
					for (size_t i = 0; i < errorIn.size(); i++)
						errorIn[i] = errorIn[i] - out.back()[i];
					// 集計
					score.Add(accumulate(errorIn.begin(), errorIn.end(), 0.0) / errorIn.size());
					// scale_pos_weight
					if (params.objective == XNNObjective::BinaryLogistic) {
						for (size_t i = 0; i < errorIn.size(); i++)
							errorIn[i] *= binaryScales[(int)round(data.out[i])];
					}
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

			return score;
		}

		// 予測
		vector<RegressionScore> Predict(const std::vector<XNNData>& testData, size_t startIndex, size_t count) const {
			vector<RegressionScore> score(params.outUnits);
#pragma omp parallel for
			for (int i = 0; i < (int)count; i++) {
				auto data = testData[startIndex + i];
				auto pred = Predict(move(data.in));
				assert((int)pred.size() == params.outUnits);
				TransformOutput(data.out);
				for (size_t o = 0; o < pred.size(); o++)
					score[o].Add(data.out[o] - pred[o]);
			}
			return score;
		}

		// 予測
		string Predict(vector<XNNData>&& testData) {
			CheckData(testData);

			// 全データ分Predictする
			vector<vector<float>> result;
			result.reserve(testData.size());
			auto startTime = high_resolution_clock::now();
			for (auto& d : testData)
				result.emplace_back(Predict(move(d.in)));
			auto milliSec = duration_cast<milliseconds>(
				high_resolution_clock::now() - startTime).count();
			auto milliSecPerPredict = (double)milliSec / testData.size();

			// 結果の集計・整形
			vector<unique_ptr<IScore>> score;
			switch (params.objective) {
			case XNNObjective::RegLinear:
			case XNNObjective::RegLogistic:
				for (int i = 0; i < params.outUnits; i++)
					score.emplace_back(new RegressionScore());
				break;
			case XNNObjective::BinaryLogistic:
				for (int i = 0; i < params.outUnits; i++)
					score.emplace_back(new ClassificationScore(2));
				break;
			case XNNObjective::MultiSoftmax:
				score.emplace_back(new ClassificationScore(params.outUnits));
				break;
			}
			stringstream raw;
			raw << fixed << setprecision(7);
			for (int i = 0; i < (int)result.size(); i++) {
				auto& data = testData[i];
				auto& pred = result[i];
				TransformOutput(data.out);
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
					cout << "--------------------------- out[" << o << "] ---------------------------" << endl;
				cout << score[o]->ToString();
				if (params.objective == XNNObjective::RegLinear ||
					params.objective == XNNObjective::RegLogistic)
					cout << endl; // TODO:そのうちなんとかする
			}

			cout << "検証完了: " << milliSecPerPredict << "ミリ秒/回" << endl;
			return raw.str();
		}

		// 予測
		vector<float> Predict(vector<float>&& in) const {
			assert((int)in.size() == params.inUnits);
			vector<float> out;
			out.reserve(params.hiddenUnits);
			for (size_t i = 0; i < layers.size(); i++) {
				layers[i]->Forward(in, out, nullptr);
				swap(in, out);
			}
			assert((int)in.size() == params.outUnits);
			return in;
		}

		// 特徴の重要度のようなもの
		vector<float> GetFScore() const {
			size_t inLayer = 0;
			if (params.scaleInput != 0)
				inLayer += 1;
			return ((FullyConnectedLayer*)layers[inLayer].get())->GetWeightRMS();
		}
	};

	vector<XNNData> LoadSVMLight(const string& path, int inUnits, int fMinIndex) {
		ifstream is(path);
		if (!is)
			throw XNNException(path + "が開けませんでした。");
		return LoadSVMLight(is, inUnits, fMinIndex);
	}
	vector<XNNData> LoadSVMLight(istream& is, int inUnits, int fMinIndex) {
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
					if (index < fMinIndex)
						throw XNNException("特徴のインデックスが" + to_string(fMinIndex) + "未満: 行の内容=" + line);
					if (inUnits + fMinIndex <= index)
						throw XNNException("特徴のインデックスが" + to_string(inUnits + fMinIndex) + "以上: 行の内容=" + line);
					data.in[index - 1] = stof(token.substr(coron + 1));
				}
			}
			result.emplace_back(move(data));
		}
		return result;
	}
	void SaveSVMLight(const string& path, const vector<XNNData>& data, int fMinIndex) {
		ofstream os(path);
		if (!os)
			throw XNNException(path + "が開けませんでした。");
		SaveSVMLight(os, data, fMinIndex);
	}
	void SaveSVMLight(ostream& os, const vector<XNNData>& data, int fMinIndex) {
		for (auto& d : data) {
			for (size_t i = 0; i < d.out.size(); i++) {
				if (0 < i)
					os << ",";
				os << d.out[i];
			}
			for (size_t i = 0; i < d.in.size(); i++)
				os << " " << (i + fMinIndex) << ":" << d.in[i];
			os << endl;
		}
	}

	struct StringTable {
		array<const char*, 4> objectives = { {
				"reg:linear",
				"reg:logistic",
				"binary:logistic",
				"multi:softmax",
			} };
		array<const char*, 5> activations = { {
				"ReLU",
				"PReLU",
				"Identity",
				"Sigmoid",
				"Softmax",
			} };
	} stringTable;

	string ToString(XNNObjective value) {
		return stringTable.objectives[(int)value];
	}
	string ToString(XNNActivation value) {
		return stringTable.activations[(int)value];
	}
	XNNObjective XNNObjectiveFromString(const string& str) {
		for (size_t i = 0; i < stringTable.objectives.size(); i++)
			if (stringTable.objectives[i] == str)
				return (XNNObjective)i;
		throw XNNException("objectiveが不正: " + str);
	}
	XNNActivation XNNActivationFromString(const string& str) {
		for (size_t i = 0; i < stringTable.activations.size(); i++)
			if (stringTable.activations[i] == str)
				return (XNNActivation)i;
		throw XNNException("activationが不正: " + str);
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
	string XNNModel::Predict(vector<XNNData>&& testData) const {
		return impl->Predict(move(testData));
	}
	vector<float> XNNModel::Predict(vector<float>&& in) const {
		return impl->Predict(move(in));
	}
	vector<float> XNNModel::GetFScore() const {
		return impl->GetFScore();
	}
}
