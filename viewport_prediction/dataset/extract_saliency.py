import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# load pretrained resnet model
model = torchvision.models.resnet50(pretrained=True)
print(model)

# define transforms to preprocess input image into format expected by model
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
# inverse transform to get normalize image back to original form for visualization
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

# transforms to resize image to the size expected by pretrained model,
# convert PIL image to tensor, and
# normalize the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,          
])

def saliency(img, model, targetdir, filename):
    '''
    get and save saliency map for input image
    '''
    # we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    input = transform(img)
    input.unsqueeze_(0)

    input.requires_grad = True
    preds = model(input)
    score, indices = torch.max(preds, 1)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    # normalize to [0..1]
    slc = (slc - slc.min())/(slc.max()-slc.min())

    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    filename = filename.split('.')[0] + '.' + 'png'
    savedir = os.path.join(targetdir, filename)
    plt.savefig(savedir, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_files_in_folder(folder_path):
    '''
    get all images file name in folder
    '''
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            files.append(filename)
    return files

def processeachsub(subdirectories):
    '''
    process each video images
    '''
    for subdir in subdirectories:
        print(subdir)
        files = get_files_in_folder(subdir)
        target_dir = 'target_dir/Wu2017_saliencyMap' # the target dir for saving saliency map
        target_path = os.path.join(target_dir, subdir.split('/')[-1])
        print(target_path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for file in files:
            img = Image.open(os.path.join(subdir, file)).convert('RGB')
            saliency(img, model, target_path, file)

if __name__ == "__main__":
    # the source dir for storing the raw video images
    subdirectories = ['source_dir/Wu2017_images/video{}_images'.format(i) for i in range(1, 10)]
    processeachsub(subdirectories)