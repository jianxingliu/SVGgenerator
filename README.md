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