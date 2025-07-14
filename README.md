<div align="center">
<h1>CLIPにおけるモーダル間の知識蒸留による構造化枝刈り</h1>
</div>


### 🏃 Installation
The code is tested on `Pytorch==1.11.0`, `cuda==11.3.1`, and `python==3.8.13`. The dependencies can be installed by:
```
conda env create -f environment.yml
```

### 使い方
作成したconda環境をアクティベートします
```
 conda activate upop
```
以下のコマンドでプログラムを実行します．（--KD をつけると枝刈り箇所の探索時に知識蒸留を使用します）
```
python3 -m torch.distributed.run --nproc_per_node=4  CLIP_compress.py --p 0.75 --epoch 9 \
--pretrained pretrained/clip_large_retrieval_coco.pth --config ./configs/retrieval_coco_clip.yaml \
--output_dir output_dir/test --KD  > output_dir/test.txt
```

### 事前準備
本手法による枝刈り箇所の探索は下流タスクで事前学習済みのモデルを対象に行います．以下のリンクよりダウンロードして /pretrained ディレクトリに保存してください．
（UPopにあるリンクも同じモデルパスです）

<div align="center">

|         |    COCO    |  Flickr30K  |
|:-------:|:----------:|:-----------:|
| モデル  | [COCO](https://drive.usercontent.google.com/download?id=10p1oPdiMUqo0MfPul5hCb_h9mCaNCh6q&export=download&authuser=0) | [Flickr30K](https://drive.usercontent.google.com/download?id=1-MZP6xQRnmLZr1_pqUK4TvOA8Ic7XCoI&export=download&authuser=0) |

</div>

本手法では学習中に枝刈り箇所を探索します．
データセットはCOCOとFlickr30Kを準備してください．
またannotationを以下のリンクよりダウンロードして解凍してください．


<div align="center">

| annotation    
|:----------:
| [link](https://drive.usercontent.google.com/download?id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD&export=download&authuser=0) |

</div>

### 実験結果 COCO dataset

**タスク：Image→Text検索 Top1性能**

|    | 枝刈り直後 | ファインチューニング後|
|:--:|:--------:|:------------------:|
|UPop|16.70|60.02|
|Ours|35.96|62.16|

**タスク：Text→Image検索 Top1性能**

|    | 枝刈り直後 | ファインチューニング後|
|:--:|:--------:|:------------------:|
|UPop|13.55|43.64|
|Ours|33.81|47.41|

### 参考にしたリポジトリ

