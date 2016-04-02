二値分類のサンプルその2
=======================

NNは出力をたくさん作れるので、
出力を3つ作って、それぞれAND/OR/XORにしてみたサンプル。

データファイルはこんな感じでカンマ区切りで。(独自拡張)

	0,0,0 1:0 2:0
	0,1,1 1:1 2:0
	0,1,1 1:0 2:1
	1,1,0 1:1 2:1


実行結果
--------

	--------------------------- out[0] ---------------------------
	class[0]: 適合率=100.0% 再現率=100.0% F値=100.0% 選択率=25.0%
	class[1]: 適合率=100.0% 再現率=100.0% F値=100.0% 選択率=75.0%
	average:  適合率=100.0% 再現率=100.0% F値=100.0%
	--------------------------- out[1] ---------------------------
	class[0]: 適合率=100.0% 再現率=100.0% F値=100.0% 選択率=75.0%
	class[1]: 適合率=100.0% 再現率=100.0% F値=100.0% 選択率=25.0%
	average:  適合率=100.0% 再現率=100.0% F値=100.0%
	--------------------------- out[2] ---------------------------
	class[0]: 適合率=100.0% 再現率=100.0% F値=100.0% 選択率=50.0%
	class[1]: 適合率=100.0% 再現率=100.0% F値=100.0% 選択率=50.0%
	average:  適合率=100.0% 再現率=100.0% F値=100.0%
