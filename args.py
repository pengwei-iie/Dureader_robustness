import torch

seed = 42
device = torch.device("cuda", 0)
test_lines = 14019  # 多少条训练数据，即：len(features), 记得修改 !!!!!!!!!!
# test_lines = 186862 # dynamic de data
search_input_file = "../data/extracted/trainset/search.train.json"
zhidao_input_file = "../data_2020/extracted/trainset/train.json"
dev_zhidao_input_file = "../data_2020/extracted/devset/dev.json"
dev_search_input_file = "../data/extracted/devset/search.dev.json"

max_seq_length = 512
max_query_length = 60

pretrained_file = "./roberta_wwm_ext"
output_dir = "./model_dir_"
predict_example_files='predict_dev.data'

max_para_num=5  # 选择几篇文档进行预测
learning_rate = 5e-5
batch_size = 4
num_train_epochs = 10
gradient_accumulation_steps = 8   # 梯度累积
num_train_optimization_steps = int(test_lines / gradient_accumulation_steps / batch_size) * num_train_epochs
log_step = int(test_lines / batch_size / 4)  # 每个epoch验证几次，默认4次
