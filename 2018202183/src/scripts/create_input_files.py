from utils import create_input_files
from imageio import imread

if __name__ == '__main__':
   # img = imread('./datasets/Flicker8k_Dataset/2513260012_03d33305cf.jpg')


    create_input_files('flickr8k', './datasets/dataset_flickr8k.json', './datasets/Flicker8k_Dataset',
                       captions_per_image=5, min_word_freq=5, output_folder='./datasets/caption_data',
                       max_len=50)

