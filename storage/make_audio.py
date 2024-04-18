from transformers import T5ForConditionalGeneration, T5Tokenizer
from pydub import AudioSegment
import torchaudio

# モデルとトークナイザーの初期化
model = T5ForConditionalGeneration.from_pretrained("sonoisa/tts-Japanese-transformer")
tokenizer = T5Tokenizer.from_pretrained("sonoisa/tts-Japanese-transformer")

# 変換したいテキスト
text = "彼は、The Wombatsのリードボーカル、ギタリスト、そして主要なソングライターとして知られています。"

# テキストをトークン化し、モデルに入力
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(inputs.input_ids)

# 音声データを取得し、音声ファイルとして保存
audio = torchaudio.transforms.GriffinLim(n_fft=2048, n_hop=256)(outputs[0].squeeze().cpu())
torchaudio.save("output.wav", audio, sample_rate=22050)
