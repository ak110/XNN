回帰のサンプル
==============

将棋の進行度を学習させてみる。
入力はKとP。出力は0～1。

ロジスティック回帰なのでラベルは[0, 1]にする必要がある。



実行結果
--------

	ネットワーク: 1858 - (16 x 1) - 1
	検証: out[0] : train={MAE= 7.3% RMSE= 9.9%} test={MAE= 7.2% RMSE= 9.9%}
	検証: out[0] : train={MAE= 7.3% RMSE= 9.8%} test={MAE= 7.4% RMSE= 9.9%}
	検証: out[0] : train={MAE= 7.2% RMSE= 9.8%} test={MAE= 7.2% RMSE= 9.8%}
	検証: out[0] : train={MAE= 7.2% RMSE= 9.8%} test={MAE= 7.2% RMSE= 9.8%}
	検証: out[0] : train={MAE= 7.2% RMSE= 9.8%} test={MAE= 7.2% RMSE= 9.8%}
	検証: out[0] : train={MAE= 7.2% RMSE= 9.7%} test={MAE= 7.2% RMSE= 9.9%}
	検証: out[0] : train={MAE= 7.2% RMSE= 9.8%} test={MAE= 7.2% RMSE= 9.9%}
	検証: out[0] : train={MAE= 7.1% RMSE= 9.6%} test={MAE= 7.1% RMSE= 9.7%}
	検証: out[0] : train={MAE= 7.2% RMSE= 9.7%} test={MAE= 7.2% RMSE= 9.8%}
	検証: out[0] : train={MAE= 7.1% RMSE= 9.7%} test={MAE= 7.2% RMSE= 9.9%}
	検証: out[0] : train={MAE= 7.2% RMSE= 9.8%} test={MAE= 7.2% RMSE= 9.8%}
	検証: out[0] : train={MAE= 7.1% RMSE= 9.7%} test={MAE= 7.2% RMSE= 9.8%}
	学習完了: 271.726秒
	保存完了: XNN.model
	====== 検証 ======
	MAE= 7.4% RMSE=10.1%
	検証完了: 0.0365031ミリ秒/回
