多クラス分類のサンプル
======================

多クラス分類の定番、MNIST。
データはここから。
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html

多クラス分類の場合、ラベルは[0, out_units)にする必要あり。

TensorFlowのチュートリアルを真似しようと思ってみたけど、
構造上2層に出来なくて3層になってしまった。
(のでちょっと正解率が高い)


実行結果
--------

	====== 学習 ======
	ネットワーク: 780+PReLU - (32+PReLU x 1) - 10
	検証: average: train={MAE= 2.1% RMSE= 9.3%} test={MAE= 2.1% RMSE= 9.5%}
	検証: average: train={MAE= 1.5% RMSE= 8.0%} test={MAE= 1.5% RMSE= 8.1%}
	検証: average: train={MAE= 1.1% RMSE= 6.8%} test={MAE= 1.2% RMSE= 7.6%}
	検証: average: train={MAE= 1.0% RMSE= 6.4%} test={MAE= 1.0% RMSE= 7.1%}
	検証: average: train={MAE= 1.0% RMSE= 6.3%} test={MAE= 1.0% RMSE= 7.1%}
	検証: average: train={MAE= 0.7% RMSE= 5.4%} test={MAE= 0.9% RMSE= 6.9%}
	検証: average: train={MAE= 0.6% RMSE= 4.9%} test={MAE= 0.9% RMSE= 6.9%}
	検証: average: train={MAE= 0.6% RMSE= 4.8%} test={MAE= 0.8% RMSE= 6.7%}
	検証: average: train={MAE= 0.6% RMSE= 5.1%} test={MAE= 0.8% RMSE= 6.8%}
	検証: average: train={MAE= 0.5% RMSE= 4.2%} test={MAE= 0.8% RMSE= 6.7%}
	検証: average: train={MAE= 0.5% RMSE= 4.7%} test={MAE= 0.8% RMSE= 6.9%}
	検証: average: train={MAE= 0.4% RMSE= 4.0%} test={MAE= 0.8% RMSE= 6.8%}
	検証: average: train={MAE= 0.5% RMSE= 4.2%} test={MAE= 0.8% RMSE= 6.7%}
	検証: average: train={MAE= 0.4% RMSE= 4.1%} test={MAE= 0.8% RMSE= 6.8%}
	検証: average: train={MAE= 0.4% RMSE= 3.7%} test={MAE= 0.7% RMSE= 6.7%}
	検証: average: train={MAE= 0.4% RMSE= 4.0%} test={MAE= 0.7% RMSE= 6.8%}
	検証: average: train={MAE= 0.4% RMSE= 3.5%} test={MAE= 0.7% RMSE= 6.8%}
	学習完了: 164.687秒
	保存完了: XNN.model
	====== 検証 ======
	class[0]: 適合率=96.4% 再現率=98.9% F値=97.6% 選択率=10.1% 分布= 9.8%
	class[1]: 適合率=98.8% 再現率=99.0% F値=98.9% 選択率=11.4% 分布=11.3%
	class[2]: 適合率=97.4% 再現率=97.3% F値=97.3% 選択率=10.3% 分布=10.3%
	class[3]: 適合率=98.4% 再現率=96.2% F値=97.3% 選択率= 9.9% 分布=10.1%
	class[4]: 適合率=97.9% 再現率=95.0% F値=96.4% 選択率= 9.5% 分布= 9.8%
	class[5]: 適合率=96.9% 再現率=96.7% F値=96.8% 選択率= 8.9% 分布= 8.9%
	class[6]: 適合率=97.9% 再現率=97.3% F値=97.6% 選択率= 9.5% 分布= 9.6%
	class[7]: 適合率=96.7% 再現率=98.1% F値=97.4% 選択率=10.4% 分布=10.3%
	class[8]: 適合率=97.0% 再現率=96.5% F値=96.8% 選択率= 9.7% 分布= 9.7%
	class[9]: 適合率=94.9% 再現率=96.9% F値=95.9% 選択率=10.3% 分布=10.1%
	average:  適合率=97.2% 再現率=97.2% F値=97.2%
	検証完了: 0.0248ミリ秒/回
