# AToMLA - Machine Learning Samples

機械学習・強化学習のサンプルコード集です。

## 📁 プロジェクト構成

- `2_1_classification.ipynb` - PyTorchによる画像分類 (MNIST)
- `3_1_svm.ipynb` - サポートベクターマシン (SVM) による分類 (Iris)
- `3_2_kmeans.ipynb` - K-means クラスタリング (Iris)
- `4_1_policy_gradient.ipynb` - 強化学習 (Policy Gradient) による迷路探索

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

### Jupyter Notebookの実行

```bash
source .venv/bin/activate  # 仮想環境を有効化
pip install jupyter  # 初回のみ
jupyter notebook --allow-root
```

ブラウザで各ノートブックファイル（`.ipynb`）を開き、セルを順に実行してください。

VS Codeで実行する場合は、Jupyter拡張機能をインストールしてノートブックを開くだけで実行できます。

## 📚 各サンプルの概要

### 2_1_classification.ipynb
- **内容**: PyTorchを使用したMNIST手書き数字の分類
- **手法**: 多層パーセプトロン (MLP)
- **モデル構成**:
  - 入力層: 784 → 256 (sigmoid)
  - 隠れ層: 256 → 128 (sigmoid)
  - ドロップアウト: 0.5
  - 出力層: 128 → 10 (softmax)
- **データセット**: MNIST (28x28ピクセルの手書き数字画像)
- **最適化**: SGD (lr=0.01), CrossEntropyLoss

### 3_1_svm.ipynb
- **内容**: サポートベクターマシンによるアヤメの分類
- **手法**: 線形カーネルSVM
- **データセット**: Iris (アヤメの花弁・がく片の測定データ)
- **可視化**: 決定境界のプロット

### 3_2_kmeans.ipynb
- **内容**: K-meansクラスタリングによるアヤメのグループ化
- **手法**: K-means (k=3)
- **データセット**: Iris
- **可視化**: クラスタリング結果の散布図

### 4_1_policy_gradient.ipynb
- **内容**: Policy Gradientによる迷路探索の学習
- **手法**: ソフトマックス方策による強化学習
- **環境**: 3x3の迷路 (状態0からゴール8へ移動)
- **可視化**: エージェントの移動アニメーション

## 🛠 必要なライブラリ

- `torch` - PyTorchディープラーニングフレームワーク
- `torchvision` - PyTorchの画像処理ライブラリ
- `numpy` - 数値計算
- `matplotlib` - グラフ描画
- `pandas` - データ処理
- `scikit-learn` - 機械学習アルゴリズム
- `mlxtend` - 決定境界の可視化
- `ipython` - ノートブック環境
- `jupyter` - Jupyter Notebook環境

詳細は `requirements.txt` を参照してください。

## 📝 ライセンス

MIT License

## 👤 Author

Richiesss
