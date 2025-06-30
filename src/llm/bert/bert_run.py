from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline

# 设置具体包含 config.json 目录
# 官方模型
model_dir=r'C:\Users\xiang\.cache\huggingface\hub\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f'
# 自定义模型
#model_dir=r'./sentiment_model'

# 加载模型和tokenizer
model=AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer=AutoTokenizer.from_pretrained(model_dir)

# 使用加载的模型和tokenizer创建分类任务的 pipeline
classifier=pipeline('text-classification',model=model,tokenizer=tokenizer,device="cuda")
# device="cuda" 和device="cpu"

# 执行分类任务
result=classifier("我非常喜欢这个电影")
print(result)
#[{'label': 'LABEL_1', 'score': 0.5223028063774109}]
#0负面  1正面 2中性

result=classifier("你好，我是AI助手")
print(result)

result=classifier("我今天很生气")
print(result)