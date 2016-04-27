回帰のサンプル
==============

将棋の進行度を学習させてみる。
入力はKとP。出力は0～1。

ロジスティック回帰なのでラベルは[0, 1]にする必要がある。



実行結果
--------

	====== 学習 ======
	ネットワーク: 1858:PReLU - (32:PReLU x 3) - 1
	検証: epoch=1 out[0] : train={MAE= 7.3% RMSE= 9.9%} test={MAE= 7.4% RMSE=10.1%}
	検証: epoch=2 out[0] : train={MAE= 7.3% RMSE= 9.8%} test={MAE= 7.3% RMSE=10.0%}
	検証: epoch=3 out[0] : train={MAE= 7.2% RMSE= 9.8%} test={MAE= 7.3% RMSE=10.1%}
	検証: epoch=4 out[0] : train={MAE= 7.2% RMSE= 9.7%} test={MAE= 7.3% RMSE=10.0%}
	検証: epoch=5 out[0] : train={MAE= 7.0% RMSE= 9.5%} test={MAE= 7.3% RMSE= 9.9%}
	検証: epoch=6 out[0] : train={MAE= 7.2% RMSE= 9.7%} test={MAE= 7.4% RMSE=10.1%}
	検証: epoch=7 out[0] : train={MAE= 7.2% RMSE= 9.7%} test={MAE= 7.4% RMSE=10.0%}
	検証: epoch=8 out[0] : train={MAE= 7.0% RMSE= 9.6%} test={MAE= 7.3% RMSE=10.0%}
	検証: epoch=9 out[0] : train={MAE= 7.1% RMSE= 9.6%} test={MAE= 7.3% RMSE=10.0%}
	学習完了: 338.098秒
	保存完了: XNN.model
	====== 検証 ======
	MAE= 7.3% RMSE= 9.9%
	検証完了: 0.0544859ミリ秒/回
