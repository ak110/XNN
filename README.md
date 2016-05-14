XNN
====
[![Build Status](https://travis-ci.org/ak110/XNN.svg?branch=master)](https://travis-ci.org/ak110/XNN)

Extremelyにしょぼい全結合NNの実装。

XGBoost風のインターフェースでお手軽に使えるものを目指して作ってみた。


特徴
----

- 今どきの技術としてはAdamとReLUだけくらいでもdeepな全結合NNが学習出来たので、基本はそんな感じに。
- 気の向くままに性能が上がりそうなものは追加していく予定。PReLUとかDropoutとか。
- CNNとかを実装するつもりはなくてただの全結合なので、画像とかに強いわけではなく。
- XNN.cppとXNN.hだけあれば動くのでC++で簡単に使いたい時に便利かも。
- GPU不使用。高速化もあんまり意識せず。
- データは全件をオンメモリに乗せてしまうので件数は程々に。
- 入力も含め全て密ベクトルで扱うので次元も程々に。


使い方
------

    XNN [<config>] [<options>]

configは省略時「"XNN.conf"」を読み込む。
optionsは「パラメータ名=値」形式でconfigの値を上書きする形で任意の個数指定可能。

例:

    XNN xxx.config task=train model_out=xxx.model

細かい使い方は[サンプル](Demo)をどうぞ。


パラメーター(モデル関連)
------------------------

既定値は出来るだけ多くの場合に通用するようなものを適当に設定。(時々変わるかも)

* objective [既定値=reg:linear]
  - reg:linear       -- 線形回帰
  - reg:logistic     -- ロジスティック回帰
  - binary:logistic  -- 2クラス分類
  - multi:softmax    -- 多クラス分類
* in_units
  - 入力のユニット数。1以上。
* hidden_units [既定値=32]
  - 隠れ層のユニット数。1以上。
* out_units [既定値=1]
  - 出力のユニット数。1以上。多クラス分類の場合はクラス数。
* hidden_layers [既定値=3]
  - 隠れ層の数。0以上。0で隠れ層無し(＝線形モデル)になる。
* activation [既定値=PReLU]
  - ReLU  -- 隠れ層の活性化関数にReLUを使用する。
  - PReLU -- 隠れ層の活性化関数にParametric ReLUを使用する。
* scale_input [既定値=true]
  - true  -- 入力をスケーリングする。(max(1,絶対値の最大値)で割るだけ。)
  - false -- 入力をスケーリングしない。
* l1 [既定値=0.0]
  - L1正則化項の重み。
* l2 [既定値=0.01]
  - L2正則化項の重み。
* dropout_keep_prob [既定値=1.0]
  - Dropoutする場合に、残す割合を(0, 1]で指定する。1ならDropoutしない。0.75なら3/4は残す。
* scale_pos_weight [既定値=-1.0]
  - 2クラス分類のときの正例(ラベルが1のデータ)の重み。1なら負例と等倍。
    0.1,10ならそれぞれ負例,正例を10倍重み付けして学習する。
	負の値を指定した場合でout_unitsが1の場合は訓練データの負例数/正例数を算出して使用する。
	負の値を指定した場合でout_unitsが1でない場合は1。


パラメーター(コマンドライン関連)
--------------------------------

* data
  - 訓練データ。
* test:data
  - 検証データ。学習時の終了判定にも使用する。
* task [既定値=all]
  - all   -- 以下の全てを順番に実施する。
  - train -- 学習する。
  - pred  -- test:dataを評価して結果の統計情報を表示する。
  - dump  -- 学習結果をname_dumpのテキストファイルに出力する。(可視化用の独自形式)
  - fscore -- 特徴の重要度のようなものを算出する。(ただの入力層の重みのRMSなので参考程度のもの)
* model_in [既定値=XNN.model]
  - pred、fscoreで使用するモデルのファイルパス
* model_out [既定値=XNN.model]
  - trainでモデルを保存するファイルパス
* name_log [既定値=XNN.log]
  - 標準出力と同様の内容を出力するログファイルのパス
* name_history [既定値=history.csv]
  - trainで学習時の損失の変化を出力するCSVファイルのパス
* name_pred [既定値=pred.txt]
  - predで結果を出力するテキストファイルのパス
* name_dump [既定値=dump.txt]
  - dumpで結果を出力するテキストファイルのパス
* name_fscore [既定値=fscore.txt]
  - fscoreで結果を出力するテキストファイルのパス
* fmap [既定値=fmap.tsv]
  - fscoreで各featureに名前を付ける場合のファイルのパス。タブ区切りでインデックスと名前を記述する。
* feature_min_index [既定値=1]
  - data/test:dataのファイルで使用する特徴のインデックスの最小値を指定する。既定値の場合は[1, n]、0を指定すると[0, n-1]になる。
* verbose [既定値=1]
  - 学習の途中経過を多めに出すなら1。(データやネットワークが大きい時用)

