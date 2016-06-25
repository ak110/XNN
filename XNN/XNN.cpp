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

	// 確率な値(0～1)の文字列化(小数点以下1桁、%単位)
	struct ProbabilityFormatter {
		int precision = 1;
		int width = 3; // 100.0%だと4だが、レアケースなのでxx.x%に合わせる
		string operator()(double value) const {
			stringstream ss;
			ss << fixed << setprecision(precision)
				<< setw(width + precision)
				<< value * 100 << "%";
			return ss.str();
		}
	};
	// 任意の実数の文字列化(有効数字3桁)
	struct PlainValueFormatter {
		int precision = 3;
		int width = 5;
		string operator()(double value) const {
			stringstream ss;
			ss << setprecision(precision)
				<< setw(width + precision)
				<< value;
			return ss.str();
		}
	};

	// 回帰の統計情報
	struct RegressionReport {
		mutex mtx;
		struct {
			double totalA = 0.0, total2 = 0.0;
			size_t count = 0;
		} values;
		RegressionReport() = default;
		// ゼロクリア
		void Clear() {
			values = remove_reference<decltype(values)>::type();
		}
		// 結果の追加
		void Add(float label, float pred) {
			double error = label - pred;
			lock_guard<mutex> lock(mtx);
			values.totalA += abs(error);
			values.total2 += error * error;
			values.count++;
		}
		// MAE(Mean absolete error、平均絶対誤差)を返す。
		double GetMAE() const { return values.totalA / values.count; }
		// RMSE(Root mean square error、二乗平均平方根誤差)を返す。
		double GetRMSE() const { return sqrt(values.total2 / values.count); }
		// シンプルな文字列化(改行なし、複数行不可)
		template<class Formatter>
		string ToString(Formatter formatter) const {
			return "MAE=" + formatter(GetMAE()) + " RMSE=" + formatter(GetRMSE());
		}
		// 詳細な文字列化(末尾に改行あり、複数行可)
		template<class Formatter>
		string ToStringDetail(Formatter formatter) const {
			return ToString(formatter) + "\n";
		}
	};
	// クラス分類の統計情報
	struct ClassificationReport {
		struct CountList : array<atomic<uint64_t>, 3> {
			CountList() = default;
			CountList(const CountList& other) {
				for (size_t i = 0; i < size(); i++)
					(*this)[i] = other[i].load();
			}
		};
		vector<CountList> list;
		ClassificationReport(int classCount = 2) : list(classCount) {}
		// ゼロクリア
		void Clear() {
			list = vector<CountList>(list.size());
		}
		// 追加(classIndex指定)
		void Add(size_t labelClass, size_t pickClass) {
			assert(labelClass < list.size());
			assert(pickClass  < list.size());
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
		// 正解率: 適合率の重み付き平均
		double GetAccuracy() const {
			uint64_t sum = 0, total = 0;
			for (size_t classIndex = 0; classIndex < list.size(); classIndex++) {
				sum += list[classIndex][0];
				total += GetLabelCount(classIndex);
			}
			return (double)sum / total;
		}
		// シンプルな文字列化(改行なし、複数行不可)
		string ToString() const {
			return "Acc=" + ProbabilityFormatter()(GetAccuracy());
		}
		// 詳細な文字列化(末尾に改行あり、複数行可)
		string ToStringDetail() const {
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

	// ニューラルネット用の統計情報
	struct IScore {
		// ゼロクリア
		virtual void Clear() = 0;
		// 結果の追加
		virtual void Add(const vector<float>& label, const vector<float>& pred) = 0;
		// MAE(Mean absolete error、平均絶対誤差)を返す。
		virtual double GetMAE() const = 0;
		// RMSE(Root mean square error、二乗平均平方根誤差)を返す。
		virtual double GetRMSE() const = 0;
		// シンプルな文字列化(改行なし、複数行不可)
		virtual string ToString() const = 0;
		// 詳細な文字列化(末尾に改行あり、複数行可)
		virtual string ToStringDetail() const = 0;
	};
	// 回帰
	template<class Formatter>
	struct RegressionScore : public IScore {
		Formatter formatter;
		vector<RegressionReport> reports;
		RegressionScore(int outCount) : reports(outCount) {}
		virtual void Clear() override {
			for (auto& s : reports)
				s.Clear();
		}
		virtual void Add(const vector<float>& label, const vector<float>& pred) override {
			assert(label.size() == reports.size());
			assert(pred.size() == reports.size());
			for (size_t i = 0, n = reports.size(); i < n; i++)
				reports[i].Add({ label[i] }, { pred[i] });
		}
		virtual double GetMAE() const override {
			// 平均を返す
			return accumulate(reports.begin(), reports.end(), 0.0,
				[](double sum, const RegressionReport& s) { return sum + s.GetMAE(); }) / reports.size();
		}
		virtual double GetRMSE() const override {
			// 平均を返す
			return accumulate(reports.begin(), reports.end(), 0.0,
				[](double sum, const RegressionReport& s) { return sum + s.GetRMSE(); }) / reports.size();
		}
		virtual string ToString() const override {
			stringstream ss;
			ss << "MAE=" << formatter(GetMAE())
				<< " RMSE=" << formatter(GetRMSE());
			return ss.str();
		}
		virtual string ToStringDetail() const override {
			stringstream ss;
			for (size_t o = 0; o < reports.size(); o++) {
				if (2 <= reports.size())
					ss << "--------------------------- out[" << o << "] ---------------------------" << endl;
				ss << reports[o].ToStringDetail(formatter);
			}
			return ss.str();
		}
	};
	// 2クラス分類
	struct BinaryClassificationScore : public IScore {
		ProbabilityFormatter formatter;
		RegressionReport regReport;
		vector<ClassificationReport> reports;
		BinaryClassificationScore(int outCount) : reports(outCount) {}
		virtual void Clear() override {
			regReport.Clear();
			for (auto& s : reports)
				s.Clear();
		}
		virtual void Add(const vector<float>& label, const vector<float>& pred) override {
			for (size_t i = 0, n = reports.size(); i < n; i++) {
				regReport.Add(label[i], pred[i]);
				reports[i].Add(size_t(0.5f <= label[i]), size_t(0.5f <= pred[i]));
			}
		}
		virtual double GetMAE() const override { return regReport.GetMAE(); }
		virtual double GetRMSE() const override { return regReport.GetRMSE(); }
		virtual string ToString() const override {
			return regReport.ToString(formatter) + " Acc=" + formatter(GetAccuracy());
		}
		virtual string ToStringDetail() const override {
			stringstream ss;
			for (size_t o = 0; o < reports.size(); o++) {
				if (2 <= reports.size())
					ss << "--------------------------- out[" << o << "] ---------------------------" << endl;
				ss << reports[o].ToStringDetail();
			}
			return ss.str();
		}
		// 正解率
		double GetAccuracy() const {
			return accumulate(reports.begin(), reports.end(), 0.0,
				[](double s, const ClassificationReport& r) { return s + r.GetAccuracy(); }) / reports.size();
		}
	};
	// 多クラス分類
	struct MulticlassClassificationScore : public IScore {
		ProbabilityFormatter formatter;
		RegressionReport regReport;
		ClassificationReport report;
		MulticlassClassificationScore(int classes) : report(classes) {}
		virtual void Clear() override {
			regReport.Clear();
			report.Clear();
		}
		virtual void Add(const vector<float>& label, const vector<float>& pred) override {
			assert(label.size() == pred.size());
			size_t labelClass = max_element(label.begin(), label.end()) - label.begin();
			size_t pickClass = max_element(pred.begin(), pred.end()) - pred.begin();
			report.Add(labelClass, pickClass);
			for (size_t i = 0, n = label.size(); i < n; i++)
				regReport.Add(label[i], pred[i]);
		}
		virtual double GetMAE() const override { return regReport.GetMAE(); }
		virtual double GetRMSE() const override { return regReport.GetRMSE(); }
		virtual string ToString() const override {
			return regReport.ToString(formatter) + " " + report.ToString();
		}
		virtual string ToStringDetail() const override {
			return report.ToStringDetail();
		}
	};

	// 統計用クラスを作成
	unique_ptr<IScore> CreateScore(const XNNParams& params) {
		switch (params.objective) {
		case XNNObjective::RegLinear:
			return unique_ptr<IScore>(new RegressionScore<PlainValueFormatter>(params.outUnits));

		case XNNObjective::RegLogistic:
			return unique_ptr<IScore>(new RegressionScore<ProbabilityFormatter>(params.outUnits));

		case XNNObjective::BinaryLogistic:
			return unique_ptr<IScore>(new BinaryClassificationScore(params.outUnits));

		case XNNObjective::MultiSoftmax:
			return unique_ptr<IScore>(new MulticlassClassificationScore(params.outUnits));

		default:
			assert(false);
			return unique_ptr<IScore>();
		}
	}

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
				x[i] = newX;
			}
			return lt;
		}
	};

	// Adamによるパラメータの更新
	struct AdamOptimizer : public OptimizerBase<AdamOptimizer> {
		unique_ptr<float[]> m, v;

		AdamOptimizer(size_t dimension, double scale) : OptimizerBase(dimension, scale), m(new float[dimension]), v(new float[dimension]) {
			fill(m.get(), m.get() + dimension, 0.0f);
			fill(v.get(), v.get() + dimension, 0.0f);
		}

		double GetStep(size_t i, double g, uint64_t lt) {
			constexpr float alpha = 0.001f, beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f;
			m[i] = beta1 * m[i] + (1.0f - beta1) * float(g);
			v[i] = beta2 * v[i] + (1.0f - beta2) * float(g) * float(g);
			auto hm = m[i] / (1.0f - (float)pow(beta1, lt));
			auto hv = v[i] / (1.0f - (float)pow(beta2, lt));
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

	// 各層の内容を文字列化するためのクラス。今のところ書式はだいぶ適当。
	struct Dumper {
		stringstream ss;
		Dumper() {
			ss << setprecision(3);
		}
		void AddLayer(const string& name) {
			ss << "layer: " << name << endl;
		}
		template<class T>
		void AddParam(const string& name, const T& value) {
			AddParam(name, to_string(value));
		}
		void AddParam(const string& name, const string& value) {
			ss << "  " << name << ": " << value << endl;
		}
		void AddParam(const string& name, const vector<float>& values) {
			AddParam(name, values.data(), 0, values.size());
		}
		void AddParam(const string& name, const float* ptr, size_t startIndex, size_t count) {
			ss << "  " << name << ": [" << endl;
			for (size_t i = 0; i < count / 8; i++) {
				ss << "   ";
				for (size_t j = 0; j < 8; j++)
					ss << " " << ptr[startIndex + i * 8 + j] << ",";
				ss << endl;
			}
			if (count % 8 != 0) {
				ss << "   ";
				for (size_t j = 0, n = count % 8; j < n; j++)
					ss << " " << ptr[startIndex + count - n + j] << ",";
				ss << endl;
			}
			ss << "  ]" << endl;
		}
		string ToString() const { return ss.str(); }
	};

	// 層
	struct ILayer {
		virtual ~ILayer() {}
		// モデルの読み込み
		virtual void Load(istream& s) {}
		// モデルの保存
		virtual void Save(ostream& s) const {}
		// モデルの文字列化
		virtual void Dump(Dumper& d) const = 0;
		// 学習用に初期化
		// 学習率を程よくするために、入力のL1ノルムの平均(の推定値)を受け取り、出力のL1ノルムの平均(の推定値)を返す。(順に伝搬させる)
		virtual void Initialize(const vector<XNNData>& data, mt19937_64& rnd, double inputNorm, double& outputNorm) = 0;
		// 学習するクラスを作る
		virtual unique_ptr<ILayerTrainer> CreateTrainer(const XNNParams& params, mt19937_64& rnd) {
			return unique_ptr<ILayerTrainer>(new NullLayerTrainer());
		}
		// 順伝搬(学習時用)
		virtual void Forward(const vector<float> in[], vector<float> out[], int batchSize, ILayerTrainer* trainer) {
#pragma omp parallel for
			for (int i = 0; i < batchSize; i++)
				Forward(in[i], out[i], trainer);
		}
		// 順伝搬(直接呼ぶのは評価時のみで、その場合trainer == nullptr。ただし学習時用はデフォルトではこれを呼ぶ。)
		virtual void Forward(const vector<float>& in, vector<float>& out, ILayerTrainer* trainer) const {
			out = in;
		}
		// 逆伝搬
		virtual void Backward(
			ILayerTrainer& trainer, int mbIndex,
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
		// モデルの文字列化
		void Dump(Dumper& d) const override {
			d.AddLayer("InputScaling");
			d.AddParam("scale", scale);
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

	// 出力をスケーリングする層。
	// 線形回帰で極端に大きい出力とかを出すのが大変なので、
	// 訓練データから平均・標準偏差を算出しておいてスケーリングしてしまう。
	struct OutputScalingLayer : public ILayer {
		uint64_t inUnits;
		vector<float> weight, bias;
		OutputScalingLayer(uint64_t inUnits) : inUnits(inUnits), weight(inUnits, 1), bias(inUnits, 0) {}
		// モデルの読み込み
		void Load(istream& s) override {
			s.read((char*)&inUnits, sizeof inUnits);
			weight.resize(inUnits);
			bias.resize(inUnits);
			s.read((char*)&weight[0], weight.size() * sizeof weight[0]);
			s.read((char*)&bias[0], bias.size() * sizeof bias[0]);
		}
		// モデルの保存
		void Save(ostream& s) const override {
			s.write((const char*)&inUnits, sizeof inUnits);
			s.write((const char*)&weight[0], weight.size() * sizeof weight[0]);
			s.write((const char*)&bias[0], bias.size() * sizeof bias[0]);
		}
		// モデルの文字列化
		void Dump(Dumper & d) const override {
			d.AddLayer("OutputScaling");
			d.AddParam("weight", weight);
			d.AddParam("bias", bias);
		}
		// 学習用に初期化
		void Initialize(const vector<XNNData>& data, mt19937_64 & rnd, double inputNorm, double & outputNorm) override {
			// 出力の平均と標準偏差を算出する
			// weight = 標準偏差 * 3
			// bias = 平均
			// とする。
			fill(weight.begin(), weight.end(), 0.0f);
			fill(bias.begin(), bias.end(), 0.0f);
			for (auto& d : data)
				bias += d.out;
			bias /= data.size(); // 平均
			for (auto& d : data)
				for (size_t i = 0; i < d.out.size(); i++)
					weight[i] = (d.out[i] - bias[i]) * (d.out[i] - bias[i]);
			for (auto& s : weight)
				s = sqrt(s / data.size() + 1e-5f) * 3; // 標準偏差 * 3
			// この層が最後の前提なので適当
			outputNorm = inputNorm;
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out, ILayerTrainer* trainer) const override {
			out = in;
			out *= weight;
			out += bias;
		}
		// 逆伝搬
		void Backward(ILayerTrainer& trainer_, int mbIndex,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const override {
			errorOut = errorIn;
			errorOut /= weight;
		}
	};

	// 全層結合層
	struct FullyConnectedLayer : public ILayer {
		uint64_t inUnits, outUnits;
		double inputNorm;
		vector<float> weights, biases;
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
		// モデルの文字列化
		void Dump(Dumper& d) const override {
			d.AddLayer("FullyConnected");
			for (size_t o = 0; o < outUnits; o++)
				d.AddParam("weights[" + to_string(o) + "]",
					weights.data(), o * inUnits, inUnits);
			d.AddParam("biases", biases);
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
				x = 0.1f; // バイアスは適当な正の定数。dead neurons対策。
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
				optimizerB(owner.biases.size(), 1),
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
		void Backward(ILayerTrainer& trainer_, int mbIndex,
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
		// モデルの文字列化
		void Dump(Dumper& d) const override {
			d.AddLayer("Activation");
			d.AddParam("function", ToString(Act));
		}
		// 学習用に初期化
		void Initialize(const vector<XNNData>& data, mt19937_64& rnd, double inputNorm, double& outputNorm) override {
			switch (Act) {
			case XNNActivation::ReLU: outputNorm = inputNorm / 2; break;
			case XNNActivation::ELU: outputNorm = inputNorm; break;
			case XNNActivation::Sigmoid: outputNorm = inputNorm / 2; break;
			case XNNActivation::Softmax: outputNorm = 1.0; break;
			}
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out, ILayerTrainer* trainer) const override {
			const float ELUAlpha = 1.0f;
			out = in;
			switch (Act) {
			case XNNActivation::ReLU:
				for (auto& x : out)
					if (x <= 0)
						x = 0.0f;
				break;
			case XNNActivation::ELU:
				for (auto& x : out)
					if (x <= 0)
						x = ELUAlpha * (exp(x) - 1);
				break;
			case XNNActivation::Sigmoid:
				for (auto& x : out)
					x = 1.0f / (1.0f + exp(-x));
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
		void Backward(ILayerTrainer& trainer, int mbIndex,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const override {
			const float ELUAlpha = 1.0f;
			errorOut = errorIn;
			assert(errorOut.size() == out.size());
			switch (Act) {
			case XNNActivation::ReLU:
				for (size_t o = 0; o < errorOut.size(); o++)
					if (in[o] <= 0)
						errorOut[o] = 0.0f;
				break;
			case XNNActivation::ELU:
				for (size_t o = 0; o < errorOut.size(); o++)
					if (in[o] <= 0)
						errorOut[o] *= ELUAlpha * (exp(in[o]) - 1) + ELUAlpha;
				break;
			case XNNActivation::Sigmoid:
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
		// モデルの文字列化
		void Dump(Dumper& d) const override {
			d.AddLayer("Activation");
			d.AddParam("function", ToString(XNNActivation::PReLU));
			d.AddParam("weights", weights);
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
		void Backward(ILayerTrainer& trainer_, int mbIndex,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const override {
			assert(in.size() == inUnits);
			assert(out.size() == inUnits);
			assert(errorIn.size() == inUnits);
			auto& trainer = (Trainer&)trainer_;
			errorOut = errorIn;
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
	// BatchNormalization
	struct BatchNormalizationLayer : public ILayer {
		const uint64_t inUnits;
		vector<float> weights;
		vector<float> biases;
		BatchNormalizationLayer(uint64_t inUnits) : inUnits(inUnits), weights(inUnits, 1.0f), biases(inUnits, 0.0f) {}
		// モデルの読み込み
		void Load(istream& s) override {
			s.read((char*)&inUnits, sizeof inUnits);
			weights.resize(inUnits);
			biases.resize(inUnits);
			s.read((char*)&weights[0], weights.size() * sizeof weights[0]);
			s.read((char*)&biases[0], biases.size() * sizeof biases[0]);
		}
		// モデルの保存
		void Save(ostream& s) const override {
			s.write((const char*)&inUnits, sizeof inUnits);
			s.write((const char*)&weights[0], weights.size() * sizeof weights[0]);
			s.write((const char*)&biases[0], biases.size() * sizeof biases[0]);
		}
		// モデルの文字列化
		void Dump(Dumper& d) const override {
			d.AddLayer("BatchNormalization");
			d.AddParam("weights", weights);
			d.AddParam("biases", biases);
		}
		// 学習用に初期化
		void Initialize(const vector<XNNData>&, mt19937_64&, double inputNorm, double& outputNorm) override {
			outputNorm = (double)inUnits;
		}
		// 学習するクラス
		struct Trainer : ILayerTrainer {
			BatchNormalizationLayer& owner;
			AdamOptimizer optimizerW;
			AdamOptimizer optimizerB;
			DenseVector<> gradW;
			DenseVector<> gradB;
			vector<float> mean, std, gamma, beta, emaMean, emaStd;
			int batchSize;
			bool firstUpdate = true;
			Trainer(BatchNormalizationLayer& owner, const XNNParams& params) :
				owner(owner),
				optimizerW(owner.weights.size(), 1),
				optimizerB(owner.biases.size(), 1),
				gradW(owner.inUnits),
				gradB(owner.inUnits),
				mean(owner.inUnits),
				std(owner.inUnits),
				gamma(owner.inUnits, 1.0f),
				beta(owner.inUnits, 0.0f),
				emaMean(owner.inUnits),
				emaStd(owner.inUnits) {
				// 正則化項
				optimizerW.l1 = params.l1;
				optimizerW.l2 = params.l2;
			}
			void Clear() override {
				gradW.Clear();
				gradB.Clear();
			}
			void Update() override {
				optimizerW.Update(gamma, gradW);
				optimizerB.Update(beta, gradB);
				// 最終結果のために平均と標準偏差の指数移動平均を算出
				if (firstUpdate) {
					firstUpdate = false;
					emaMean = mean;
					emaStd = std;
				} else {
					const float momentum = 0.9f;
					for (size_t i = 0; i < owner.inUnits; i++) {
						emaMean[i] = momentum * emaMean[i] + (1 - momentum) * mean[i];
						emaStd[i] = momentum * emaStd[i] + (1 - momentum) * std[i];
					}
				}
				// 評価用のweight/biasの算出
				for (size_t i = 0; i < owner.inUnits; i++) {
					owner.weights[i] = gamma[i] / emaStd[i];
					owner.biases[i] = beta[i] - owner.weights[i] * emaMean[i];
				}
			}
		};
		// 学習するクラスを作る
		unique_ptr<ILayerTrainer> CreateTrainer(const XNNParams& params, mt19937_64& rnd) override {
			return unique_ptr<ILayerTrainer>(new Trainer(*this, params));
		}
		// 順伝搬(学習時用)
		void Forward(const vector<float> in[], vector<float> out[], int batchSize, ILayerTrainer* trainer_) override {
			auto& trainer = (Trainer&)*trainer_;
			trainer.batchSize = batchSize;
			// 平均
			fill_n(trainer.mean.begin(), inUnits, 0.0f);
			for (int mb = 0; mb < batchSize; mb++)
				for (size_t i = 0; i < inUnits; i++)
					trainer.mean[i] += in[mb][i];
			trainer.mean /= batchSize;
			// 標準偏差
			fill_n(trainer.std.begin(), inUnits, 0.0f);
			for (int mb = 0; mb < batchSize; mb++)
				for (size_t i = 0; i < inUnits; i++) {
					auto d = in[mb][i] - trainer.mean[i];
					trainer.std[i] += d * d;
				}
			trainer.std /= batchSize;
			for (size_t i = 0; i < inUnits; i++)
				trainer.std[i] = sqrt(trainer.std[i] + 1e-6f);
			// 評価用のweight/biasの算出
			// gamma * (元の値 - mean) / std + beta
			// = {gamma / std} * (元の値 - mean) + beta
			// = {gamma / std} * 元の値 + {gamma / std} * (- mean) + beta
			// = {gamma / std} * 元の値 + {beta - {gamma / std} * mean}
			for (size_t i = 0; i < inUnits; i++) {
				weights[i] = trainer.gamma[i] / trainer.std[i];
				biases[i] = trainer.beta[i] - weights[i] * trainer.mean[i];
			}
			// 順伝搬
			ILayer::Forward(in, out, batchSize, trainer_);
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out, ILayerTrainer* trainer) const override {
			assert(in.size() == inUnits);
			out = in;
			out *= weights;
			out += biases;
		}
		// 逆伝搬
		void Backward(ILayerTrainer& trainer_, int mbIndex,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const override {
			assert(in.size() == inUnits);
			assert(out.size() == inUnits);
			assert(errorIn.size() == inUnits);
			auto& trainer = (Trainer&)trainer_;
			errorOut.clear();
			errorOut.resize(inUnits);
			for (size_t i = 0; i < inUnits; i++) {
				// 誤差の伝播
				auto x = in[i] - trainer.mean[i];
				auto gx = errorIn[i] * trainer.gamma[i];
				auto gs = gx * x * (-0.5f) * pow(trainer.std[i], -3);
				auto gm = gx * (-1) / trainer.std[i] + gs * (-2) * x / trainer.batchSize;
				errorOut[i] =
					gx / trainer.std[i] +
					gs * 2 * x / trainer.batchSize +
					gm / trainer.batchSize;
				// gammaの勾配
				auto hx = x / trainer.std[i];
				trainer.gradW += make_pair(i, errorIn[i] * hx);
				// betaの勾配
				trainer.gradB += make_pair(i, errorIn[i]);
			}
		}
	};
	// Dropout
	struct DropoutLayer : public ILayer {
		const uint64_t inUnits;
		const float keepProb;
		DropoutLayer(uint64_t inUnits, float keepProb) : inUnits(inUnits), keepProb(keepProb) {}
		// モデルの文字列化
		void Dump(Dumper& d) const override {
			d.AddLayer("Dropout");
			d.AddParam("keepProb", keepProb);
		}
		// 学習用に初期化
		void Initialize(const vector<XNNData>&, mt19937_64&, double inputNorm, double& outputNorm) override {
			outputNorm = inputNorm / 2;
		}
		struct Trainer : public NullLayerTrainer {
			mt19937_64 rnd;
			vector<unique_ptr<bool[]>> keepFlags;
			Trainer(mt19937_64& rnd, uint64_t inUnits) : rnd(rnd()) {}
		};
		// 学習するクラスを作る
		unique_ptr<ILayerTrainer> CreateTrainer(const XNNParams& params, mt19937_64& rnd) {
			return unique_ptr<ILayerTrainer>(new Trainer(rnd, inUnits));
		}
		// 順伝搬(学習時用)
		void Forward(const vector<float> in[], vector<float> out[], int batchSize, ILayerTrainer* trainer_) override {
			auto& trainer = (Trainer&)*trainer_;
			trainer.keepFlags.resize(batchSize);
			for (int mb = 0; mb < batchSize; mb++) {
				if (!trainer.keepFlags[mb])
					trainer.keepFlags[mb].reset(new bool[inUnits]);
				auto keepFlags = trainer.keepFlags[mb].get();
				auto keepCount = (size_t)round(inUnits * keepProb);
				for (size_t i = 0; i < keepCount; i++)
					keepFlags[i] = true;
				for (size_t i = keepCount; i < inUnits; i++)
					keepFlags[i] = false;
				shuffle(keepFlags, keepFlags + inUnits, trainer.rnd);
			}

#pragma omp parallel for
			for (int mb = 0; mb < batchSize; mb++) {
				out[mb] = in[mb];
				for (size_t i = 0; i < inUnits; i++)
					if (trainer.keepFlags[mb][i])
						out[mb][i] /= keepProb; // keep
					else
						out[mb][i] = 0; // drop
			}
		}
		// 順伝搬
		void Forward(const vector<float>& in, vector<float>& out, ILayerTrainer* trainer) const override {
			assert(in.size() == inUnits);
			out = in;
		}
		// 逆伝搬
		void Backward(ILayerTrainer& trainer_, int mbIndex,
			const vector<float>& in, const vector<float>& out,
			vector<float>& errorOut, const vector<float>& errorIn) const override {
			assert(in.size() == inUnits);
			assert(out.size() == inUnits);
			assert(errorIn.size() == inUnits);
			auto& trainer = (Trainer&)trainer_;
			errorOut = errorIn;
			for (size_t i = 0; i < inUnits; i++) {
				if (trainer.keepFlags[mbIndex][i])
					errorOut[i] /= keepProb; // keep
				else
					errorOut[i] = 0; // drop
			}
		}
	};

	// モデル(バイナリファイル)のヘッダ
	// XNNParamsの内容をファイルに記録するための形式
#pragma pack(push, 1)
	struct XNNModelHeader {
		array<char, 4> signature = { { 'X', 'N' , 'N' , 'M' } };
		uint32_t version = 1;

		typedef array<char, 16> CharArray16; // マクロのカンマ対策
#define ITEMS(F) \
	F(CharArray16, objective, { {} }) \
	F(int32_t, inUnits, 0) \
	F(int32_t, hiddenUnits, 0) \
	F(int32_t, hiddenLayers, 0) \
	F(int32_t, outUnits, 0) \
	F(CharArray16, activation, { {} }) \
	F(int32_t, scaleInput, 0) \
	F(int32_t, batchNormalization, 0) \
	F(float, l1, 0) \
	F(float, l2, 0) \
	F(float, dropoutKeepProb, 0) \
	F(float, scalePosWeight, 0) \
	F(int32_t, verbose, 0) \
	F(int32_t, minEpoch, 0) \
	F(int32_t, maxEpoch, 0) \
	F(int32_t, miniBatchSize, 0) \
	F(int32_t, minMiniBatchCount, 0) \
	F(int32_t, earlyStoppingTolerance, 0) \
	// 項目一覧のマクロ
#define F(type, name, init) type name = init;
		ITEMS(F)
#undef F

		array<int32_t, 102> reserved = {};

		XNNModelHeader() = default;
		void OnLoad() {
			if (memcmp(signature.data(), "XNNM", 4) != 0)
				throw XNNException("未対応のファイル形式です。");
			// 今後バージョン間の互換性を保つための処理を追加する場合は以下へ。
		}
		void operator<<(const XNNParams& params) {
#define F(type, name, init) Read(name, params.name);
			ITEMS(F);
#undef F
		}
		void operator>>(XNNParams& params) const {
#define F(type, name, init) Write(name, params.name);
			ITEMS(F);
#undef F
		}

	private:
		void Read(int32_t& x, int y) { x = y; }
		void Read(int32_t& x, bool y) { x = y ? 1 : 0; }
		void Read(float& x, float y) { x = y; }
		void Read(CharArray16& x, XNNObjective y) { strcpy(x.data(), ToString(y).c_str()); }
		void Read(CharArray16& x, XNNActivation y) { strcpy(x.data(), ToString(y).c_str()); }
		void Write(int32_t x, int& y) const { y = x; }
		void Write(int32_t x, bool& y) const { y = x != 0; }
		void Write(float x, float& y) const { y = x; }
		void Write(CharArray16 x, XNNObjective& y) const { y = XNNObjectiveFromString(x.data()); }
		void Write(CharArray16 x, XNNActivation& y) const { y = XNNActivationFromString(x.data()); }
#undef ITEMS
	};
#pragma pack(pop)
	static_assert(sizeof(XNNModelHeader) == 4 * 128, "sizeof(XNNModelHeader)");

	// 標準出力とファイルにログ出力するためのクラス
	struct Logger {
		ostream* log = nullptr;
		template<class T>
		const Logger& operator<<(const T& value) const {
			cout << value;
			if (log != nullptr)
				*log << value;
			return *this;
		}
		const Logger& operator<<(ostream& (* value)(ostream&)) const {
			cout << value;
			if (log != nullptr)
				*log << value;
			return *this;
		}
	};
	// 実装。
	struct XNNModel::XNNImpl {
		XNNParams params;
		vector<unique_ptr<ILayer>> layers;
		vector<unique_ptr<ILayerTrainer>> trainers;
		mt19937_64 mt;
		ostream* history = nullptr;
		Logger logger;

		XNNImpl(const XNNParams& params) : params(params) {
			Initialize();
		}
		XNNImpl(const string& path) : params(0, 0, 0, 0) { Load(path); }

		void SetLog(ostream& os) {
			logger.log = &os;
		}
		void SetHistory(ostream& os) {
			history = &os;
		}
		void Save(const string& path) const {
			ofstream fs(path, ios_base::binary);
			Save(fs);
		}
		void Save(ostream& fs) const {
			XNNModelHeader header;
			header << params;
			fs.write((const char*)&header, sizeof header);
			for (auto& l : layers)
				l->Save(fs);
		}
		void Load(const string& path) {
			ifstream fs(path, ios_base::binary);
			if (!fs)
				throw XNNException(path + "が開けませんでした。");
			try {
				Load(fs);
			} catch (exception& e) {
				throw XNNException(path + "の読み込み失敗: " + e.what());
			}
		}
		void Load(istream& fs) {
			XNNModelHeader header;
			fs.read((char*)&header, sizeof header);
			header.OnLoad();
			header >> params;
			Initialize();
			for (auto& l : layers)
				l->Load(fs);
		}
		string Dump() const {
			Dumper d;
			for (auto& l : layers)
				l->Dump(d);
			return d.ToString();
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
				else if (params.activation == XNNActivation::ELU)
					layers.emplace_back(new ActivationLayer<XNNActivation::ELU>(outUnits));
				else
					throw XNNException("activationが不正: " + ToString(params.activation));
				if (params.batchNormalization != 0)
					layers.emplace_back(new BatchNormalizationLayer(outUnits));
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
					layers.emplace_back(new OutputScalingLayer(outUnits));
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
				logger << "訓練データ: " << trainData.size() << "件" << endl;
				logger << "検証データ: " << testData.size() << "件" << endl;
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
				logger << "scale_pos_weight = " << params.scalePosWeight << endl;
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

			// ミニバッチに各クラスが均等に含まれるように調整する。
			// (2クラス分類で出力が1つだけの場合と、多クラス分類のみ。)
			if ((params.objective == XNNObjective::BinaryLogistic && params.outUnits == 1) ||
				params.objective == XNNObjective::MultiSoftmax) {
				// 各クラスの情報
				auto classCount = params.objective == XNNObjective::BinaryLogistic ? 2 : params.outUnits;
				struct ClassInfo { size_t picking, pickingFrac, picked = 0, total = 0; };
				vector<ClassInfo> classes(classCount);
				// 各クラスの個数を調べる
				for (auto& d : trainData) {
					assert(d.out.size() == 1);
					classes[(int)round(d.out[0])].total++;
				}
				// 各ミニバッチの補正処理
				auto miniBatchCount = trainData.size() / (size_t)params.miniBatchSize;
				for (size_t mb = 0; mb < miniBatchCount; mb++) {
					auto offset = mb * (size_t)params.miniBatchSize;
					auto offsetEnd = offset + (size_t)params.miniBatchSize;
					// 各クラスで今回のミニバッチに入れる個数を算出(端数切捨て)
					for (int c = 0; c < classCount; c++) {
						auto s = offsetEnd * classes[c].total -
							classes[c].picked * trainData.size();
						classes[c].picking = s / trainData.size();
						classes[c].pickingFrac = s % trainData.size(); // 端数
					}
					// 切り捨てなので足りない場合があるので、その場合は最も端数が大きいものを1個増やす
					while (true) {
						size_t n = accumulate(classes.begin(), classes.end(), 0ull,
							[](size_t x, const ClassInfo& ci) { return x + ci.picking; });
						assert((int)n <= params.miniBatchSize);
						if (params.miniBatchSize <= (int)n)
							break;
						// 最も端数の大きい物を選んで1個増やす
						auto it = max_element(classes.begin(), classes.end(),
							[](const ClassInfo& x, const ClassInfo& y) {
							return less<size_t>()(x.pickingFrac, y.pickingFrac);
						});
						it->picking++;
						it->pickingFrac = 0;
					}

					// 整列
					size_t skipCount = 0;
					for (size_t o = offset; o < offsetEnd; o++) {
						while (true) {
							int c = (int)round(trainData[o].out[0]);
							if (0 < classes[c].picking) {
								classes[c].picked++;
								classes[c].picking--;
								break;
							}
							swap(trainData[o], trainData[offsetEnd + skipCount]);
							skipCount++;
						}
					}
				}
			}

			// 設定の確認のためネットワークの大きさを表示
			logger << "ネットワーク: " << params.inUnits
				<< ":" << ToString(params.activation)
				<< " - (" << params.hiddenUnits
				<< ":" << ToString(params.activation)
				<< " x " << params.hiddenLayers
				<< ") - " << params.outUnits
				<< endl;
			if (1 <= params.verbose)
				logger << "学習開始" << endl;

			// 学習を回す。検証誤差がほぼ下がらなくなったら終了。

			double minRMSE = numeric_limits<double>::max();
			int earlyStoppingCount = 0;
			stringstream bestModel;

			size_t testSize = min(max(testData.size(), (size_t)10000), trainData.size());

			for (int epoch = 1; ; epoch++) {
				// 学習
				{
					ProgressTimer timer;

					const size_t MaxMB = trainData.size() / params.miniBatchSize;
					auto score = CreateScore(params);
					for (size_t mb = 0; mb < MaxMB; mb++) {
						PartialFit(trainData, mb * params.miniBatchSize, params.miniBatchSize, *score);

						if (1 <= params.verbose && (mb + 1) % (MaxMB / 10) == 0) {
							timer.Set(mb + 1, MaxMB);
							logger << "学習:"
								<< " epoch=" << epoch
								<< " " << timer.ToStringCount()
								<< " train={" << score->ToString() << "}"
								<< " " << timer.ToStringTime()
								<< endl;
							score->Clear();
						}
					}
				}

				// 訓練誤差・検証誤差の算出
				shuffle(trainData.begin(), trainData.begin() + testSize, mt);
				auto pred1 = Predict(trainData, 0, testSize);
				auto pred2 = Predict(testData, 0, testData.size());

				// 表示
				logger << "検証: epoch=" << epoch
					<< " train={" << pred1->ToString() << "}"
					<< " test={" << pred2->ToString() << "}"
					<< endl;
				if (history != nullptr) {
					if (epoch == 1)
						*history << "epoch,train mae,train rmse,test mae,test rmse" << endl;
					*history << epoch
						<< "," << pred1->GetMAE()
						<< "," << pred1->GetRMSE()
						<< "," << pred2->GetMAE()
						<< "," << pred2->GetRMSE()
						<< endl;
				}

				// 終了判定
				auto rmse = pred2->GetRMSE();
				if (rmse < 0.005)
					break; // 充分小さければ止まる
				if (rmse < minRMSE) {
					// 最高記録を更新
					minRMSE = rmse;
					earlyStoppingCount = 0;
					// モデルを保存
					bestModel.str(""); // 空にする
					Save(bestModel);
					// maxEpoch以上ならそこで終了
					if (params.minEpoch <= epoch &&
						params.maxEpoch != 0 &&
						params.maxEpoch <= epoch)
						break;
				} else {
					++earlyStoppingCount;
					if ((params.maxEpoch != 0 && params.maxEpoch <= epoch) ||
						(params.minEpoch <= epoch &&
							params.earlyStoppingTolerance < earlyStoppingCount)) {
						// 最高記録のモデルを復元
						Load(bestModel);
						break;
					}
				}

				// シャッフル
				shuffle(trainData.begin(), trainData.end(), mt);
			}

			auto dt = high_resolution_clock::now() - startTime;
			logger << "学習完了: " << (duration_cast<milliseconds>(dt).count() / 1000.0) << "秒" << endl;
		}
		// ミニバッチによる更新
		void PartialFit(const std::vector<XNNData>& trainData, size_t startIndex, size_t count, IScore& score) {
			// 勾配をクリア
#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < (int)trainers.size(); i++)
				trainers[i]->Clear();

			// scale_pos_weightの比率を保ちつつ、あまり極端な値にならないスケールを算出
			// 例: scale_pos_weight=1なら{ 1, 1 }、10なら{ 0.1818, 1.818 }。
			const array<float, 2> binaryScales = { {
				2 / (params.scalePosWeight + 1),
				params.scalePosWeight * 2 / (params.scalePosWeight + 1),
			} };

			// 順伝搬
			vector<vector<vector<float>>> out(layers.size() + 1); // layer×miniBatch×feats
			for (auto& b : out)
				b.resize(count);
			for (size_t mb = 0; mb < count; mb++)
				out[0][mb] = trainData[startIndex + mb].in;
			for (size_t i = 0; i < layers.size(); i++)
				layers[i]->Forward(out[i].data(), out[i + 1].data(), (int)count, trainers[i].get());

			// 逆伝搬
			struct ThreadLocal {
				vector<float> errorIn, errorOut;
				ThreadLocal(int hiddenUnits) : errorIn(hiddenUnits), errorOut(hiddenUnits) {}
			};
			vector<ThreadLocal> locals(omp_get_max_threads(), ThreadLocal(params.hiddenUnits));
#pragma omp parallel for
			for (int mb = 0; mb < (int)count; mb++) {
				auto& data = trainData[startIndex + mb];
				auto& local = locals[omp_get_thread_num()];
				auto& errorIn = local.errorIn;
				auto& errorOut = local.errorOut;
				// 集計
				errorIn = data.out;
				TransformOutput(errorIn);
				score.Add(errorIn, out.back()[mb]);
				// エラーを算出
				// ロジスティック回帰／線形二乗誤差：教師 - 予測
				for (size_t i = 0; i < errorIn.size(); i++)
					errorIn[i] = errorIn[i] - out.back()[mb][i];
				// scale_pos_weight
				if (params.objective == XNNObjective::BinaryLogistic) {
					for (size_t i = 0; i < errorIn.size(); i++)
						errorIn[i] *= binaryScales[(int)round(data.out[i])];
				}
				// 逆伝搬
				for (int i = (int)layers.size() - 1; 0 <= i; i--) {
					layers[i]->Backward(*trainers[i], mb,
						out[i][mb], out[i + 1][mb], errorOut, errorIn);
					swap(errorIn, errorOut);
				}
			}

			// 重みを更新
#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < (int)trainers.size(); i++)
				trainers[i]->Update();
		}

		// 予測
		unique_ptr<IScore> Predict(const std::vector<XNNData>& testData, size_t startIndex, size_t count) const {
			unique_ptr<IScore> score = CreateScore(params);
#pragma omp parallel for
			for (int i = 0; i < (int)count; i++) {
				auto data = testData[startIndex + i];
				auto pred = Predict(move(data.in));
				assert((int)pred.size() == params.outUnits);
				TransformOutput(data.out);
				score->Add(data.out, pred);
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
			unique_ptr<IScore> score = CreateScore(params);
			stringstream raw;
			raw << fixed << setprecision(7);
			for (int i = 0; i < (int)result.size(); i++) {
				auto& data = testData[i];
				auto& pred = result[i];
				TransformOutput(data.out);
				score->Add(data.out, pred);
				for (size_t o = 0; o < pred.size(); o++) {
					if (0 < o)
						raw << " ";
					raw << pred[o];
				}
				raw << endl;
			}
			logger << score->ToStringDetail() << flush;
			logger << "検証完了: " << milliSecPerPredict << "ミリ秒/回" << endl;
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
					data.in[index - fMinIndex] = stof(token.substr(coron + 1));
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
				"ELU",
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

	void XNNModel::SetLog(ostream& os) {
		impl->SetLog(os);
	}
	void XNNModel::SetHistory(ostream& os) {
		impl->SetHistory(os);
	}
	void XNNModel::Save(const string& path) const {
		impl->Save(path);
	}
	void XNNModel::Load(const string& path) {
		impl->Load(path);
	}
	string XNNModel::Dump() const {
		return impl->Dump();
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
