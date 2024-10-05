# dataloader/preprocessors.py

def default_preprocess(sample):
    # Implement default preprocessing
    sample = sample * 2
    return sample

def normalize(sample):
    # Get the min and max values to normalize the list
    return (sample/255)

def augment(sample):
    # Implement data augmentation logic
    return sample
