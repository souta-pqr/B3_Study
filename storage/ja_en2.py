import os
import pysbd
from transformers import pipeline
from tqdm import tqdm  # プログレスバーを表示するためのライブラリ

# 翻訳モデルの設定
fugu_translator = pipeline('translation', model='staka/fugumt-en-ja')

# output_textディレクトリのパス
input_dir = "output_text"

# output_japanese_textディレクトリを作成
output_dir = "output_japanese_text"
os.makedirs(output_dir, exist_ok=True)

# サブディレクトリの総数を取得
total_subdirs = len(os.listdir(input_dir))

# output_textディレクトリ内のサブディレクトリを処理
for i, sub_dir_name in enumerate(os.listdir(input_dir), start=1):
    sub_dir_path = os.path.join(input_dir, sub_dir_name)

    # output_japanese_textディレクトリ内にサブディレクトリを作成
    output_sub_dir_path = os.path.join(output_dir, sub_dir_name)
    os.makedirs(output_sub_dir_path, exist_ok=True)

    # サブディレクトリ内のファイルを処理
    files = os.listdir(sub_dir_path)
    for filename in tqdm(files, desc=f"Processing subdirectory {i}/{total_subdirs}", unit="file"):
        file_path = os.path.join(sub_dir_path, filename)

        # ファイルを読み込む
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 英語の文章を1文ずつに分割
        seg_en = pysbd.Segmenter(language="en", clean=False)
        sentences_en = seg_en.segment(content)

        # 英語から日本語に翻訳
        translations = fugu_translator(sentences_en)

        # 翻訳された内容をファイルに保存
        output_file_path = os.path.join(output_sub_dir_path, filename)
        with open(output_file_path, "w", encoding="utf-8") as f:
            for translation in translations:
                f.write(translation['translation_text'] + "\n")

print("翻訳が完了しました。")