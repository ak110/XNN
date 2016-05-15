二値分類のサンプルその1
=======================

パーセプトロンといえばXOR、ということで、XORを学習。

データファイルはSVMlight形式。

	0 1:0 2:0
	1 1:1 2:0
	1 1:0 2:1
	0 1:1 2:1

ラベルは0または1に。

訓練データは数が少なくても大丈夫。(整数倍して繰り返し使う)


入力
----

- [設定ファイル(XNN.conf)](XNN.conf)
- [データファイル(data.train)](data.train)
- [データファイル(data.test)](data.test) (サンプルなのでdata.trainと中身は同じ)


実行結果
--------

- [ログファイル(XNN.log)](XNN.log)
- [学習時の損失の変化(history.csv)](history.csv)
- [学習後にdata.testを評価した結果(pred.txt)](pred.txt)
- [学習結果のモデルの内容(dump.txt)](dump.txt)
- [入力の重要度のようなもの(fscore.txt)](fscore.txt)

ロジスティック回帰の結果が0.5以下なら0、0.5を超えるなら1として、
それぞれ正解した割合などを集計しています。

averageの再現率がいわゆる正解率(Accuracy)に該当します。
