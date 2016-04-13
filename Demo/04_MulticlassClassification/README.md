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
	ネットワーク: 780 - (32 x 1) - 10
	検証: average: train={MAE= 1.9% RMSE= 9.0%} test={MAE= 2.0% RMSE= 9.2%}
	検証: average: train={MAE= 1.5% RMSE= 8.0%} test={MAE= 1.4% RMSE= 8.1%}
	検証: average: train={MAE= 1.2% RMSE= 7.0%} test={MAE= 1.3% RMSE= 7.7%}
	検証: average: train={MAE= 1.0% RMSE= 6.7%} test={MAE= 1.1% RMSE= 7.1%}
	検証: average: train={MAE= 1.0% RMSE= 6.2%} test={MAE= 1.0% RMSE= 6.9%}
	検証: average: train={MAE= 0.8% RMSE= 5.6%} test={MAE= 0.9% RMSE= 6.7%}
	検証: average: train={MAE= 0.7% RMSE= 5.3%} test={MAE= 0.9% RMSE= 6.6%}
	検証: average: train={MAE= 0.6% RMSE= 4.9%} test={MAE= 0.8% RMSE= 6.4%}
	検証: average: train={MAE= 0.7% RMSE= 5.2%} test={MAE= 0.8% RMSE= 6.5%}
	検証: average: train={MAE= 0.6% RMSE= 4.7%} test={MAE= 0.8% RMSE= 6.5%}
	検証: average: train={MAE= 0.6% RMSE= 4.8%} test={MAE= 0.8% RMSE= 6.5%}
	検証: average: train={MAE= 0.5% RMSE= 4.4%} test={MAE= 0.8% RMSE= 6.5%}
	学習完了: 156.32秒
	保存完了: XNN.model
	====== 検証 ======
	class[0]: 適合率=97.0% 再現率=99.0% F値=98.0% 選択率=10.0%
	class[1]: 適合率=97.9% 再現率=98.9% F値=98.4% 選択率=11.5%
	class[2]: 適合率=96.2% 再現率=97.9% F値=97.0% 選択率=10.5%
	class[3]: 適合率=97.1% 再現率=97.5% F値=97.3% 選択率=10.1%
	class[4]: 適合率=97.8% 再現率=96.7% F値=97.3% 選択率= 9.7%
	class[5]: 適合率=98.4% 再現率=95.6% F値=97.0% 選択率= 8.7%
	class[6]: 適合率=97.6% 再現率=97.8% F値=97.7% 選択率= 9.6%
	class[7]: 適合率=97.8% 再現率=95.3% F値=96.6% 選択率=10.0%
	class[8]: 適合率=96.8% 再現率=97.1% F値=97.0% 選択率= 9.8%
	class[9]: 適合率=96.1% 再現率=96.4% F値=96.3% 選択率=10.1%
	average:  適合率=97.3% 再現率=97.3% F値=97.3%
	検証完了: 0.0245ミリ秒/回
