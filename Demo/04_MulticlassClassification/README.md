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
	検証: average: train={MAE= 2.0% RMSE= 9.0%} test={MAE= 2.1% RMSE= 9.4%}
	検証: average: train={MAE= 1.5% RMSE= 8.0%} test={MAE= 1.5% RMSE= 7.9%}
	検証: average: train={MAE= 1.2% RMSE= 6.9%} test={MAE= 1.2% RMSE= 7.3%}
	検証: average: train={MAE= 1.0% RMSE= 6.6%} test={MAE= 1.1% RMSE= 6.8%}
	検証: average: train={MAE= 0.9% RMSE= 6.0%} test={MAE= 0.9% RMSE= 6.1%}
	検証: average: train={MAE= 0.8% RMSE= 5.5%} test={MAE= 0.8% RMSE= 5.8%}
	検証: average: train={MAE= 0.7% RMSE= 5.2%} test={MAE= 0.8% RMSE= 5.6%}
	検証: average: train={MAE= 0.7% RMSE= 5.2%} test={MAE= 0.7% RMSE= 5.2%}
	検証: average: train={MAE= 0.6% RMSE= 4.8%} test={MAE= 0.6% RMSE= 5.2%}
	学習完了: 99.591秒
	保存完了: XNN.model
	====== 検証 ======
	class[0]: 適合率=97.5% 再現率=98.3% F値=97.9% 選択率= 9.9%
	class[1]: 適合率=98.9% 再現率=98.8% F値=98.8% 選択率=11.3%
	class[2]: 適合率=95.9% 再現率=97.8% F値=96.8% 選択率=10.5%
	class[3]: 適合率=96.7% 再現率=97.2% F値=96.9% 選択率=10.2%
	class[4]: 適合率=97.6% 再現率=96.5% F値=97.1% 選択率= 9.7%
	class[5]: 適合率=96.6% 再現率=96.7% F値=96.7% 選択率= 8.9%
	class[6]: 適合率=97.9% 再現率=97.0% F値=97.4% 選択率= 9.5%
	class[7]: 適合率=97.3% 再現率=97.2% F値=97.2% 選択率=10.3%
	class[8]: 適合率=95.0% 再現率=96.7% F値=95.8% 選択率= 9.9%
	class[9]: 適合率=98.0% 再現率=94.9% F値=96.4% 選択率= 9.8%
	average:  適合率=97.1% 再現率=97.1% F値=97.1%
	検証完了: 0.0296ミリ秒/回

