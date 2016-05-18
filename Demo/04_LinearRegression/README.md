回帰のサンプル
==============

objective = reg:linearで線形回帰。出力は任意の実数。
(ただしあまりちゃんと対応していないので表示などはだいぶ適当。)

損失関数は二乗誤差。


入力
----

- [設定ファイル(XNN.conf)](XNN.conf)
- [データファイル(data.train)](data.train)
- [データファイル(data.test)](data.test)

検証データを訓練データより細かくしたので、
pred.txtをそのままグラフにすると、
どう補間されるのか可視化出来る。
(活性化関数の影響が大きそう。)


実行結果
--------

- [ログファイル(XNN.log)](XNN.log)
- [学習時の損失の変化(history.csv)](history.csv)
- [学習後にdata.testを評価した結果(pred.txt)](pred.txt)
- [学習結果のモデルの内容(dump.txt)](dump.txt)
- [入力の重要度のようなもの(fscore.txt)](fscore.txt)

