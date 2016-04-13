
#include "XNN.h"
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>
#include <unordered_map>

using namespace XNN;

int Usage() {
	cerr << "使い方: XNN <config> [<options>]" << endl;
	cerr << "  <options>  xxx=yyy形式で、configの項目を上書き出来ます。" << endl;
	cerr << endl;
	return 1;
}

struct Config {
	unordered_map<string, string> config;
	void Add(const string& name, const string& value) {
		config[boost::trim_copy(name)] = boost::trim_copy(value);
	}
	string Get(const string& name, const string& defaultValue = string()) const {
		auto it = config.find(name);
		if (it != config.end())
			return it->second;
		return defaultValue;
	}
	const string& GetRequired(const string& name) const {
		auto it = config.find(name);
		if (it != config.end())
			return it->second;
		throw XNNException("コンフィグファイルおよびコマンドラインで " + name + " が指定されていません。");
	}
	bool GetBool(const string& name, bool defaultValue) const {
		auto s = Get(name, defaultValue ? "true" : "false");
		if (s == "true")
			return true;
		if (s == "false")
			return false;
		throw XNNException(name + "の値が不正: " + s);
	}
	XNNParams CreateParams() const {
		XNNParams params(
			stoi(GetRequired("in_units")),
			stoi(GetRequired("hidden_units")),
			stoi(GetRequired("out_units")),
			stoi(GetRequired("hidden_layers")));

		auto objective = Get("objective", "reg:logistic");
		if (objective == "reg:logistic")
			params.objective = XNNObjective::RegLogistic;
		else if (objective == "binary:logistic")
			params.objective = XNNObjective::BinaryLogistic;
		else if (objective == "multi:softmax")
			params.objective = XNNObjective::MultiSoftmax;
		else
			throw XNNException("objectiveの値が不正: " + objective);

		params.scaleInput = GetBool("scale_input", true) ? 1 : 0;
		params.verbose = stoi(Get("verbose", "1"));
		params.scalePosWeight = stod(Get("scale_pos_weight", "-1.0"));
		return params;
	}
};

int Process(int argc, char* argv[]) {
	string confPath;
	unordered_map<string, string> argConfig;
	Config config;
	// コマンドラインの取得
	{
		vector<string> args(argv + 1, argv + argc);
		for (auto& a : args) {
			auto eq = a.find('=');
			if (eq != string::npos)
				argConfig[boost::trim_copy(a.substr(0, eq))] =
				boost::trim_copy(a.substr(eq + 1));
			else if (confPath != "")
				return Usage();
			else
				confPath = a;
		}
	}
	if (confPath == "")
		return Usage();
	// configの読み込み
	{
		ifstream ifs(confPath);
		if (!ifs)
			throw XNNException("コンフィグファイル読み込み失敗。ファイル=" + confPath);
		string line;
		for (int lineCount = 1; getline(ifs, line); lineCount++) {
			if (line.length() <= 0 || line[0] == '#')
				continue; // 空行orコメント
			auto eq = line.find('=');
			if (eq == string::npos)
				throw XNNException("コンフィグファイル読み込み失敗。行=" + to_string(lineCount));
			config.Add(line.substr(0, eq), line.substr(eq + 1));
		}
		// 引数の値で上書き
		for (auto& p : argConfig)
			config.Add(p.first, p.second);
	}

	auto task = config.Get("task", "train");
	if (task == "train") {
		auto params = config.CreateParams();
		auto modelPath = config.Get("model_out", "XNN.model");
		XNNModel dnn(params);
		dnn.Train(
			LoadSVMLight(config.GetRequired("data"), params.inUnits),
			LoadSVMLight(config.GetRequired("test:data"), params.inUnits));
		dnn.Save(modelPath);
		cout << "保存完了: " << modelPath << endl;
	} else if (task == "pred") {
		auto params = config.CreateParams();
		auto modelPath = config.Get("model_in", "XNN.model");
		auto predPath = config.Get("name_pred", "pred.txt");
		XNNModel dnn(modelPath);
		auto r = dnn.Predict(LoadSVMLight(config.GetRequired("test:data"), params.inUnits));
		ofstream(predPath) << r;
	} else {
		throw XNNException("taskの値が不正。task=" + task);
	}
	return 0;
}

int main(int argc, char* argv[]) {
	try {
		return Process(argc, argv);
	} catch (std::exception& e) {
		cerr << "エラー: " << e.what() << endl;
		return 1;
	}
}

