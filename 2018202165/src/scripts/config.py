import os

"""
The config class is using to set hyper parameters
"""


class Config():
    num_filters = 50
    window_sizes = [2, 3, 4]
    num_batches = 8000  # Number of batches to train
    num_batches_batch_loss = 50  # Number of batches to show loss
    # Number of batches to check loss and accuracy on validation dataset
    num_batches_val_loss_and_acc = 300
    num_batches_save_checkpoint = 400  # Number of batches to save check point
    batch_size = 256
    learning_rate = 0.001
    train_validation_split = (0.8, 0.2)  # The ratio splitting the dataset into training set and validation set
    num_workers = 1  # Number of workers for data loading
    num_clicked_news_a_user = 10  # Number of sampled click history for each user
    # Whether try to load checkpoint
    load_checkpoint = os.environ[
        'LOAD_CHECKPOINT'] == '1' if 'LOAD_CHECKPOINT' in os.environ else True
    num_words_a_news = 10  
    entity_confidence_threshold = 0.5

    word_freq_threshold = 3
    entity_freq_threshold = 3

    # Modify this by the output of `src/dataprocess.py`
    num_word_tokens = 1 + 14760
    # Modify the following only if you use another dataset
    word_embedding_dim = 100
    entity_embedding_dim = 100
