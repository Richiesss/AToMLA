# AToMLA - Machine Learning Samples

機械学習・強化学習のサンプルコード集です。

## 📁 プロジェクト構成

- `2_1_classification.py` - ニューラルネットワークによる画像分類 (MNIST)
- `3_1_svm.py` - サポートベクターマシン (SVM) による分類 (Iris)
- `3_2_kmeans.py` - K-means クラスタリング (Iris)
- `4_1_policy_gradient.py` - 強化学習 (Policy Gradient) による迷路探索
- `4_1_policy_gradient.ipynb` - 強化学習のノートブック版（アニメーション表示対応）

## 🚀 セットアップ

### 1. リポジトリのクローン

```bash
git clone <your-repo-url>
cd AToMLA
```

### 2. 仮想環境の作成とライブラリのインストール

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 💻 実行方法

### Pythonスクリプトの実行

```bash
source .venv/bin/activate  # 仮想環境を有効化

# ニューラルネットワーク (MNIST分類)
python 2_1_classification.py

# SVM (Iris分類)
python 3_1_svm.py

# K-means (Irisクラスタリング)
python 3_2_kmeans.py

# 強化学習 (迷路探索)
python 4_1_policy_gradient.py
```

### Jupyter Notebookの実行

```bash
source .venv/bin/activate
pip install jupyter  # 初回のみ
jupyter notebook --allow-root
```

ブラウザで `4_1_policy_gradient.ipynb` を開き、セルを順に実行するとアニメーションが表示されます。

## 📚 各サンプルの概要

### 2_1_classification.py
- **内容**: TensorFlow/Kerasを使用したMNIST手書き数字の分類
- **手法**: 多層パーセプトロン (MLP)
- **データセット**: MNIST (28x28ピクセルの手書き数字画像)

### 3_1_svm.py
- **内容**: サポートベクターマシンによるアヤメの分類
- **手法**: 線形カーネルSVM
- **データセット**: Iris (アヤメの花弁・がく片の測定データ)
- **可視化**: 決定境界のプロット

### 3_2_kmeans.py
- **内容**: K-meansクラスタリングによるアヤメのグループ化
- **手法**: K-means (k=3)
- **データセット**: Iris
- **可視化**: クラスタリング結果の散布図

### 4_1_policy_gradient.py / .ipynb
- **内容**: Policy Gradientによる迷路探索の学習
- **手法**: ソフトマックス方策による強化学習
- **環境**: 3x3の迷路 (状態0からゴール8へ移動)
- **可視化**: エージェントの移動アニメーション (ノートブック版)

## 🛠 必要なライブラリ

- `numpy` - 数値計算
- `matplotlib` - グラフ描画
- `pandas` - データ処理
- `scikit-learn` - 機械学習アルゴリズム
- `mlxtend` - 決定境界の可視化
- `tensorflow` - ディープラーニング
- `ipython` - ノートブック環境

詳細は `requirements.txt` を参照してください。

## 📝 ライセンス

MIT License

## 👤 Author

Richiesss
