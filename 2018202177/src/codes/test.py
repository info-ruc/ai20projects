from inference import *

img_path = 'datasets/Flicker8k_Dataset/42637987_866635edf6.jpg'
image = Image.open(img_path).convert("RGB")
caps, cap_scores, alphas = get_image_caption(image)

visualization_att(image, caps, alphas)