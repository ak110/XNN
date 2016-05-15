多クラス分類のサンプル
======================

objective = multi:softmaxで多クラス分類。

out_unitsにクラス数を設定する。

データのラベルは[0, out_units)にする必要あり。


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

