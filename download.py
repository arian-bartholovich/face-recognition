import kagglehub

#Used to download training data

path = kagglehub.dataset_download("yilingliu610/donald-trump-face-images")

print("Path to dataset files:", path)