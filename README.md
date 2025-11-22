# AIGC

自用的 AIGC 相关的一些脚本。基本上都是 AI (主要是 Gemini 2.5 / 3.0 Pro) 写的。

## GPT-SoVITS 相关

* audios_cluster_speechbrain.py : 使用 SpeechBrain 对所有 *.wav 文件根据说话人自带分类。参数为输入文件夹、输出文件夹。 `python audios_cluster_speechbrain.py voices voices-clustered`. 
* audios_transcribe.py : 使用 litagin/anime-whisper 模型生成所有 *.wav 文件的转写文本 *.txt 文件。参数为音频文件夹。``
* gpt_sovits_evaluation_batch_inference.py : 适用于 [GPT-SoVITS][] 项目。批量使用保存的 GPT 和 SoVITS 模型 checkpoints 生成音频，用于评估不同 epoch 模型的效果。
* gpt_sovits_visqol_batch_eval.py : 使用 [ViSQOL][] 对批量生成的音频和原始音频进行质量评分并选出 Best 3 供人工试听比对。
* visqol : 静态编译的 visqol-3.3.3 Linux amd64 可执行文件。需要配合从 ViSQOL 仓库下载的[模型](https://github.com/google/visqol/tree/master/model)使用。

### gpt_sovits_evaluation_batch_inference.py

It must be put into GPT-SoVITS project root dir (docker: `/workspace/GPT-SoVITS`).

```
python evaluation_batch_inference.py --ref /data/kotomi-evaluation/z4421_01700 --input /data/kotomi-evaluation/input --output /data/kotomi-evaluation/output --version v4 --project kotomi --gpt-epoch "1-100"   --sovits-epoch "1-100"
```

- Reference audio & text: `/data/kotomi-evaluation/z4421_01700.wav` & `/data/kotomi-evaluation/z4421_01700.txt`
- Input texts: `/data/kotomi-evaluation/input/*.txt`
- Output audios in `/data/kotomi-evaluation/output/` dir: `<project>_<version>_gpt<gpt_epoch>_sovits<sovits_epoch>_<input_filename>.wav`. E.g. `kotomi_v4_gpt024_sovits057_z4419_00220.wav`

### gpt_sovits_visqol_batch_eval.py

假设原始音频(input/*.wav) 是 48k。

评估 GPT-SoVITS v4 模型生成的音频 (48k):

```
python gpt_sovits_visqol_batch_eval.py --visqol-model-path /data/visqol-model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite --gen-audio-dir /data/kotomi-evaluation/output --ref-audio-dir /data/kotomi-evaluation/input --candidates-dir /data/kotomi-evaluation/candidates
```

评估 GPT-SoVITS v2 模型生成的音频 (32k):

visqol 需要评估的音频和原始音频是相同采样率，需要先将 `input/*.wav` 的所有 48k 原始音频用 ffmpeg 转为 32k，生成 `input/<filename>_32k.wav` 版本。

Linux bash:

```bash
for f in *.wav; do [[ "$f" == *"_32k.wav"* ]] || ffmpeg -y -i "$f" -ar 32000 "${f%.wav}_32k.wav"; done
```

The above single-line Linux bash command to convert all *.wav files to 32kHz, while ignoring files already named *_32k.wav as input, but silently overwriting any existing output files.

然后调用脚本评估音频：

```
python gpt_sovits_visqol_batch_eval.py --visqol-model-path /data/visqol-model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite --gen-audio-dir /data/kotomi-evaluation/output --ref-audio-dir /data/kotomi-evaluation/input --candidates-dir /data/kotomi-evaluation/candidates --ref-audio-suffix _32k
```

输出评估结果到 stdout。示例输出： gpt_sovits_visqol_batch_eval_output_example.txt。

[GPT-SoVITS]: https://github.com/RVC-Boss/GPT-SoVITS
[ViSQOL]: https://github.com/google/visqol