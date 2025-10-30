from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

model_name = "distilbert-base-uncased"
save_directory = "./distilbert-local"

# 下载并保存分词器
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)

# 下载并保存模型
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.save_pretrained(save_directory)

print(f"模型和分词器已成功保存至 '{save_directory}' 文件夹")