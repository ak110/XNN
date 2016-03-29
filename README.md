XNN
====

Extremelyにしょぼい全結合NNの実装。

XGBoost風のインターフェースでお手軽に使えるものを目指して作ってみた。


特徴
----

- 使っている今どきの技術はAdamとReLUくらい。AutoEncoderやらRBMやらは未使用。
  こんなんでもdeepなNNが学習出来ます。
- とはいえCNNとかを実装するつもりはなくてただの全結合なので、画像とかに強いわけではなく。
- XNN.cppとXNN.hだけあれば動くのでC++で簡単に使いたい時に便利かも。
- GPU不使用。高速化もあんまり意識せず。
- データは全件をオンメモリに乗せてしまうので件数は程々に。
- 入力も含め全て密ベクトルで扱うので次元も程々に。


使い方
------

    XNN <config> [<options>]

例:

    XNN xxx.config task=train model_out=xxx.model

細かい使い方は[サンプル](Demo)をどうぞ。


パラメーター(モデル関連)
------------------------

* objective [既定値=reg:logistic]
  - reg:logistic     -- ロジスティック回帰
  - binary:logistic  -- 2クラス分類
  - multi:softmax    -- 多クラス分類
* in_units
  - 入力のユニット数。1以上。
* hidden_units
  - 隠れ層のユニット数。1以上。
* out_units
  - 出力のユニット数。1以上。多クラス分類の場合はクラス数。
* hidden_layers
  - 隠れ層の数。1以上。1でいわゆる3層NN。0以下は未対応。
* scale_input [既定値=true]
  - true  -- 入力をスケーリングする。(max(1,絶対値の最大値)で割るだけ。)
  - false -- 入力をスケーリングしない。

パラメーター(コマンドライン関連)
--------------------------------

* data
  - 訓練データ。
* test:data
  - 検証データ。
* task [既定値=train]
  - train -- 学習する。
  - pred  -- test:dataを評価して結果の統計情報を表示する。
* model_in [既定値=XNN.model]
  - predで使用するモデルのファイルパス
* model_out [既定値=XNN.model]
  - trainでモデルを保存するファイルパス
* name_pred [既定値=pred.txt]
  - predで結果を出力するテキストファイルのパス
* verbose [既定値=1]
  - 学習の途中経過を多めに出すなら1。(データやネットワークが大きい時用)

