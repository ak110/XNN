回帰のサンプル
==============

将棋の進行度を学習させてみる。
入力はKとP。出力は0～1。

ロジスティック回帰なのでラベルは[0, 1]にする必要がある。



実行結果
--------

	====== 学習 ======
	ネットワーク: 1858 - (16 x 1) - 1
	検証: out[0] : train={MAE= 7.3% RMSE= 9.9%} test={MAE= 7.4% RMSE=10.1%}
	検証: out[0] : train={MAE= 7.3% RMSE=10.0%} test={MAE= 7.4% RMSE=10.1%}
	検証: out[0] : train={MAE= 7.2% RMSE= 9.7%} test={MAE= 7.3% RMSE=10.0%}
	検証: out[0] : train={MAE= 7.3% RMSE= 9.8%} test={MAE= 7.5% RMSE=10.1%}
	検証: out[0] : train={MAE= 7.2% RMSE= 9.8%} test={MAE= 7.4% RMSE=10.0%}
	検証: out[0] : train={MAE= 7.3% RMSE= 9.8%} test={MAE= 7.4% RMSE=10.1%}
	検証: out[0] : train={MAE= 7.2% RMSE= 9.9%} test={MAE= 7.4% RMSE=10.2%}
	学習完了: 177.61秒
	保存完了: XNN.model
	====== 検証 ======
	MAE= 7.4% RMSE=10.2%
	検証完了: 0.0294664ミリ秒/回
