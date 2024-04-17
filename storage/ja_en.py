import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm  # プログレスバーを表示するためのライブラリ
import concurrent.futures

# 翻訳モデルとトークナイザーの設定
model_name = 'sonoisa/t5-base-english-japanese'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# output_textディレクトリのパス
input_dir = "output_text"

# output_japanese_textディレクトリを作成
output_dir = "output_japanese_text"
os.makedirs(output_dir, exist_ok=True)

# サブディレクトリの総数を取得
total_subdirs = len(os.listdir(input_dir))

def translate_file(filename, sub_dir_path, output_sub_dir_path):
    file_path = os.path.join(sub_dir_path, filename)

    # ファイルを読み込む
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 英語から日本語に翻訳
    inputs = tokenizer.encode("translate English to Japanese: " + content, return_tensors="pt")
    outputs = model.generate(inputs, max_length=600, num_beams=4, early_stopping=True)
    translation = tokenizer.decode(outputs[0])

    # 翻訳された内容をファイルに保存
    output_file_path = os.path.join(output_sub_dir_path, filename)
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(translation)

# output_textディレクトリ内のサブディレクトリを処理
for i, sub_dir_name in enumerate(os.listdir(input_dir), start=1):
    sub_dir_path = os.path.join(input_dir, sub_dir_name)

    # output_japanese_textディレクトリ内にサブディレクトリを作成
    output_sub_dir_path = os.path.join(output_dir, sub_dir_name)
    os.makedirs(output_sub_dir_path, exist_ok=True)

    # サブディレクトリ内のファイルを処理
    files = os.listdir(sub_dir_path)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(translate_file, files, [sub_dir_path]*len(files), [output_sub_dir_path]*len(files)), total=len(files), desc=f"Processing subdirectory {i}/{total_subdirs}", unit="file"))

print("翻訳が完了しました。")
