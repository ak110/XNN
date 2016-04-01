
#include "XNN.h"
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>
#include <unordered_map>

using namespace XNN;

int Usage() {
	cerr << "�g����: XNN <config> [<options>]" << endl;
	cerr << "  <options>  xxx=yyy�`���ŁAconfig�̍��ڂ��㏑���o���܂��B" << endl;
	cerr << endl;
	return 1;
}

struct Config {
	unordered_map<string, string> config;
	void Add(const string& name, const string& value) {
		config[boost::trim_copy(name)] = boost::trim_copy(value);
	}
	const string& Get(const string& name, const string& defaultValue = string()) const {
		auto it = config.find(name);
		if (it != config.end())
			return it->second;
		return defaultValue;
	}
	const string& GetRequired(const string& name) const {
		auto it = config.find(name);
		if (it != config.end())
			return it->second;
		throw XNNException("�R���t�B�O�t�@�C������уR�}���h���C���� " + name + " ���w�肳��Ă��܂���B");
	}
	bool GetBool(const string& name, bool defaultValue) const {
		auto s = Get(name, defaultValue ? "true" : "false");
		if (s == "true")
			return true;
		if (s == "false")
			return false;
		throw XNNException(name + "�̒l���s��: " + s);
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
			throw XNNException("objective�̒l���s��: " + objective);

		params.scaleInput = GetBool("scale_input", true) ? 1 : 0;
		params.verbose = stoi(Get("verbose", "1"));
		return params;
	}
};

int Process(int argc, char* argv[]) {
	string confPath;
	unordered_map<string, string> argConfig;
	Config config;
	// �R�}���h���C���̎擾
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
	// config�̓ǂݍ���
	{
		ifstream ifs(confPath);
		if (!ifs)
			throw XNNException("�R���t�B�O�t�@�C���ǂݍ��ݎ��s�B�t�@�C��=" + confPath);
		string line;
		for (int lineCount = 1; getline(ifs, line); lineCount++) {
			if (line.length() <= 0 || line[0] == '#')
				continue; // ��sor�R�����g
			auto eq = line.find('=');
			if (eq == string::npos)
				throw XNNException("�R���t�B�O�t�@�C���ǂݍ��ݎ��s�B�s=" + to_string(lineCount));
			config.Add(line.substr(0, eq), line.substr(eq + 1));
		}
		// �����̒l�ŏ㏑��
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
		cout << "�ۑ�����: " << modelPath << endl;
	} else if (task == "pred") {
		auto params = config.CreateParams();
		auto modelPath = config.Get("model_in", "XNN.model");
		auto predPath = config.Get("name_pred", "pred.txt");
		XNNModel dnn(modelPath);
		auto r = dnn.Predict(LoadSVMLight(config.GetRequired("test:data"), params.inUnits));
		cout << r.statistics << endl;
		ofstream(predPath) << r.raw;
	} else {
		throw XNNException("task�̒l���s���Btask=" + task);
	}
	return 0;
}

int main(int argc, char* argv[]) {
	try {
		return Process(argc, argv);
	} catch (std::exception& e) {
		cerr << "�G���[: " << e.what() << endl;
		return 1;
	}
}

