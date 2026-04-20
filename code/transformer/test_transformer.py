# Transformers 快速上手
参考: https://huggingface.co/docs/transformers/main/en/quicktour

## 方式一：Pipeline (最简单，推荐入门)

from transformers import pipeline

# 创建情感分析 pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 推理
result = classifier("I love this movie!")
print(result)
# 输出: [{'label': 'POSITIVE', 'score': 0.9998}]

# 批量推理
results = classifier(["I love this", "This is terrible", "Not bad at all"])
print(results)

## 方式二：AutoClass (更灵活)

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# 自动加载模型和 tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 方式二推理
inputs = tokenizer("I love this book!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
# 获取预测结果
import torch
pred = torch.argmax(outputs.logits, dim=1).item()
label = model.config.id2label[pred]
print(f"预测: {label}")

## 方式三：Text Generation (文本生成)

generator = pipeline("text-generation", model="distilbert-base-uncased")
results = generator("Once upon a time", max_length=30, num_return_sequences=2)
print(results)