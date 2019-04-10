network_type = 'Transformer'  #Avaible: Transformar and DAN
path_to_data = '/mnt/data/danil.akhmetov/PycharmProjects/RuBERT/data/date=2019-04-05.csv'
path_to_test_data ='/mnt/data/danil.akhmetov/PycharmProjects/EasyNetNLP/data/fresh.parquet'
path_to_save_model = "output/model/" + network_type
path_to_load_model = "output/model/" + network_type + '_epoch_1.h5'
max_seq_length = 64
batch_size = 512
learning_rate = 1e-3
n_epochs = 10