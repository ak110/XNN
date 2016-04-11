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
	検証: average: train={MAE= 2.0% RMSE= 9.1%} test={MAE= 2.1% RMSE= 9.4%}
	検証: average: train={MAE= 1.5% RMSE= 7.8%} test={MAE= 1.6% RMSE= 8.1%}
	検証: average: train={MAE= 1.2% RMSE= 7.0%} test={MAE= 1.3% RMSE= 7.5%}
	検証: average: train={MAE= 1.0% RMSE= 6.3%} test={MAE= 1.1% RMSE= 6.9%}
	検証: average: train={MAE= 0.9% RMSE= 6.0%} test={MAE= 0.9% RMSE= 6.2%}
	検証: average: train={MAE= 0.8% RMSE= 5.5%} test={MAE= 0.9% RMSE= 5.9%}
	検証: average: train={MAE= 0.8% RMSE= 5.5%} test={MAE= 0.8% RMSE= 5.9%}
	検証: average: train={MAE= 0.6% RMSE= 4.7%} test={MAE= 0.7% RMSE= 5.5%}
	検証: average: train={MAE= 0.7% RMSE= 5.2%} test={MAE= 0.8% RMSE= 5.8%}
	検証: average: train={MAE= 0.6% RMSE= 4.8%} test={MAE= 0.6% RMSE= 5.1%}
	検証: average: train={MAE= 0.6% RMSE= 4.5%} test={MAE= 0.6% RMSE= 5.0%}
	検証: average: train={MAE= 0.5% RMSE= 4.3%} test={MAE= 0.6% RMSE= 4.8%}
	検証: average: train={MAE= 0.5% RMSE= 4.4%} test={MAE= 0.6% RMSE= 4.8%}
	検証: average: train={MAE= 0.5% RMSE= 4.3%} test={MAE= 0.5% RMSE= 4.6%}
	検証: average: train={MAE= 0.5% RMSE= 4.4%} test={MAE= 0.5% RMSE= 4.7%}
	検証: average: train={MAE= 0.5% RMSE= 4.2%} test={MAE= 0.5% RMSE= 4.6%}
	検証: average: train={MAE= 0.4% RMSE= 3.9%} test={MAE= 0.4% RMSE= 4.1%}
	検証: average: train={MAE= 0.5% RMSE= 4.2%} test={MAE= 0.5% RMSE= 4.5%}
	検証: average: train={MAE= 0.4% RMSE= 3.3%} test={MAE= 0.4% RMSE= 3.9%}
	検証: average: train={MAE= 0.4% RMSE= 3.7%} test={MAE= 0.4% RMSE= 4.0%}
	検証: average: train={MAE= 0.4% RMSE= 4.1%} test={MAE= 0.5% RMSE= 4.5%}
	検証: average: train={MAE= 0.4% RMSE= 3.6%} test={MAE= 0.4% RMSE= 4.1%}
	検証: average: train={MAE= 0.3% RMSE= 3.0%} test={MAE= 0.4% RMSE= 3.8%}
	検証: average: train={MAE= 0.3% RMSE= 3.3%} test={MAE= 0.4% RMSE= 3.9%}
	検証: average: train={MAE= 0.3% RMSE= 3.3%} test={MAE= 0.4% RMSE= 3.7%}
	検証: average: train={MAE= 0.3% RMSE= 3.1%} test={MAE= 0.3% RMSE= 3.3%}
	検証: average: train={MAE= 0.3% RMSE= 3.2%} test={MAE= 0.4% RMSE= 3.5%}
	検証: average: train={MAE= 0.4% RMSE= 3.5%} test={MAE= 0.4% RMSE= 4.0%}
	検証: average: train={MAE= 0.3% RMSE= 3.1%} test={MAE= 0.3% RMSE= 3.4%}
	検証: average: train={MAE= 0.3% RMSE= 3.3%} test={MAE= 0.4% RMSE= 3.5%}
	学習完了: 343.812秒
	保存完了: XNN.model
	====== 検証 ======
	class[0]: 適合率=97.3% 再現率=98.4% F値=97.8% 選択率= 9.9%
	class[1]: 適合率=98.8% 再現率=98.5% F値=98.6% 選択率=11.3%
	class[2]: 適合率=97.8% 再現率=96.4% F値=97.1% 選択率=10.2%
	class[3]: 適合率=98.1% 再現率=96.3% F値=97.2% 選択率= 9.9%
	class[4]: 適合率=97.1% 再現率=97.6% F値=97.3% 選択率= 9.9%
	class[5]: 適合率=97.5% 再現率=95.9% F値=96.7% 選択率= 8.8%
	class[6]: 適合率=97.1% 再現率=98.0% F値=97.6% 選択率= 9.7%
	class[7]: 適合率=95.4% 再現率=97.6% F値=96.5% 選択率=10.5%
	class[8]: 適合率=95.7% 再現率=96.7% F値=96.2% 選択率= 9.8%
	class[9]: 適合率=96.3% 再現率=95.6% F値=96.0% 選択率=10.0%
	average:  適合率=97.1% 再現率=97.1% F値=97.1%
	検証完了: 0.0292ミリ秒/回
