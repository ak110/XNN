二値分類のサンプルその2
=======================

NNは出力をたくさん作れるので、
出力を3つ作って、それぞれAND/OR/XORにしてみたサンプル。

データファイルはこんな感じでカンマ区切りで。(独自拡張)

	0,0,0 1:0 2:0
	0,1,1 1:1 2:0
	0,1,1 1:0 2:1
	1,1,0 1:1 2:1


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

