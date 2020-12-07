import os
import numpy as np

lst_files = os.listdir("./images/")
lst_images = []

for file in lst_files:
  if ".txt" not in file:
    lst_images.append(file)
    
# Validation set
random_idx = np.random.randint(0, len(lst_images), 1000)

with open("./darknet/train.txt","w") as f:
  for idx in range(len(lst_images)):
    if idx not in random_idx:
      f.write("data/images/"+lst_images[idx]+"\n")
      
with open("./darknet/val.txt","w") as f:
    for idx in random_idx:
      f.write("data/images/"+lst_images[idx]+"\n")
