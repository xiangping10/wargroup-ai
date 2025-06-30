# 步骤一：环境准备
# pip install torch transformers dataset scikit-learn

# 步骤二：加载中文BERT预训练模型
from transformers import BertTokenizer,BertForSequenceClassification

# 加载BERT中文预训练模型和分词器
tokenizers=BertTokenizer.from_pretrained('bert-base-chinese')
model=BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    ignore_mismatched_sizes=True,
    num_labels=3)
# num_labels=3:表示我们要进行三类情感分类(正面、负面、中性)

# 步骤三：加载ChnSentiCorp数据集
from datasets import load_dataset

# 数据集地址:https://huggingface.co/datasets/lansinuote/ChnSentiCorp
dataset=load_dataset('lansinuote/ChnSentiCorp')

import re

# 定义数据清洗函数
def clean_text(text):
    text=re.sub(r'[^\w\s]','',text)
    # 去掉前后空格
    text=text.strip()
    return text

# 对数据集中的文本进行清洗
dataset=dataset.map(lambda x:{'text':clean_text(x['text'])})

# 步骤四： 数据预处理
def tokenize_function(examples):
    return tokenizers(examples['text'],truncation=True,padding='max_length',max_length=128)

# 对数据集进行分词和编码
encoded_dataset=dataset.map(tokenize_function,batched=True)

# 步骤五： 训练模型
from transformers import Trainer,TrainingArguments

# 定义训练参数，创建一个TrainingArguments对象
training_args=TrainingArguments(
    # 指定训练输出的目录，用于保存模型和其他输出文件
    output_dir='./results',
    # 设置训练的轮数
    num_train_epochs=1,
    # 设置每个GPU的训练批次大小
    per_device_train_batch_size=1,
    # 设置每个GPU的评估批次大小
    per_device_eval_batch_size=1,
    # 设置评估策略为每个epoch结束后进行评估
    eval_strategy='epoch',
    # 指定日志保存目录
    logging_dir='./logs'
)

# 创建一个Trainer对象，并传入模型、训练参数和数据集
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation']
)

# 开始训练
#trainer.train()
#{'loss': 1.0312, 'grad_norm': 2.7945194244384766, 'learning_rate': 4.4796875e-05, 'epoch': 0.1}

# 步骤六： 模型评估
from sklearn.metrics import accuracy_score

# 定义一个函数，用于计算准确率
def compute_metrics(p):
    preds=p.predictions.argmax(-1)
    return {'accuracy':accuracy_score(p.label_ids,preds)}

# 在测试集上评估模型
trainer.evaluate(encoded_dataset['test'],metric_key_prefix="eval")
#{'eval_loss': 0.7085, 'eval_accuracy': 0.5, 'eval_precision': 0.5, 'eval_recall': 0.5, 'eval_f1': 0.5, 'eval_epoch': 0.1}
# eval_loss 损失函数
# eval_accuracy 准确率

# 步骤七：导出模型
model.save_pretrained('./sentiment_model')
tokenizers.save_pretrained('./sentiment_model')