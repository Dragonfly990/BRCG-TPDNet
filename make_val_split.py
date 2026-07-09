import os, random

random.seed(4222)  

train_img_dir = 'data/CVC-ClinicDB/train/images/'
val_list_path = 'data/CVC-ClinicDB/val_list.txt'

all_imgs = sorted([f for f in os.listdir(train_img_dir) 
                   if f.endswith(('.png', '.jpg'))])

val_imgs = random.sample(all_imgs, 60)

with open(val_list_path, 'w') as f:
    for name in val_imgs:
        f.write(name + '\n')

print(f"Saved {len(val_imgs)} val images to {val_list_path}")
print("First 5:", val_imgs[:5])


