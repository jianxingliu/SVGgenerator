# SVGgenerator
### SVG annotation
对SVG图像进行标注，使用模型：DeepSeek-VL-7B-Chat
```
cd DeepSeek-VL
python annotation.py
```
标注结果保存在`annotation`文件夹
### Score the annotation
对于每一张图片，随机选取9张与其具有不同特征的其他图片，利用大模型读入标与注所有10张图片，选取best match，若match成功则此标注score + 1.
使用模型： Qwen2-VL-7B-Instruct 
```
cd ..
python qwenvl2.py
```
得分结果保存在`annotation_score`文件夹

Note:为防止原始PNG文件太大导致占用显存多，运行速度慢，可以
```
python smallPng.py
```
缩小PNG图片。

目前得分结果概览：
```
Score 0: 42 SVGs
Score 1: 38 SVGs
Score 2: 52 SVGs
Score 3: 56 SVGs
Score 4: 70 SVGs
Score 5: 111 SVGs
Score 6: 154 SVGs
Score 7: 329 SVGs
Score 8: 774 SVGs
Score 9: 2309 SVGs
Score 10: 4890 SVGs
8825 SVGs in total
```