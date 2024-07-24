import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
import os


tensor_dict = {}


def extract_vit_features(img):
    '''
    Extract features using Vision Transformer (ViT).
    Reference: https://discuss.pytorch.org/t/feature-extraction-in-torchvision-models-vit-b-16/148029
    '''

    model = torchvision.models.vit_b_16(pretrained=True)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    conv = feature_extractor[0]  

    # This is the whole encoder sequence
    encoder = feature_extractor[1]

    # The output shape is the one desired 
    x = model._process_input(img)

    n = x.shape[0]
    # Expand the class token to the full batch
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = encoder(x)
    x = x[:, 0]
    return x

def get_number_of_files(folder_path):
    file_count = 0
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

def processeachsub(subdirectories, start_folder):
    folder_number = start_folder
    for subdir in subdirectories:
        print(subdir)
        count = get_number_of_files(subdir)
        target_dir = '/data/data1/wangxianda/get_multimodal/Wu2017_features'
        target_path = os.path.join(target_dir, subdir.split('/')[-1])
        print(target_path)

        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for file in range(1, count+1):
            store_feature(os.path.join(subdir, f'{file}.png'), file, count, tensor_dict, folder_number)

        folder_number = folder_number + 1

def store_feature(img_dir, n, count, tensor_dict, folder_number):
    '''
    storing features in a dictionary
    '''
    img = Image.open(img_dir).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # to 224x224 
        transforms.ToTensor()  # change to tensor
    ])
    img_tensor = preprocess(img)
    img = img_tensor.unsqueeze(0)
    raw_feature = extract_vit_features(img)
    tensor_dict[f'{n}'] = raw_feature
    # save the features per 100 images
    if n % 100 == 0:   
        torch.save(tensor_dict, f'target_dir/Wu2017_features/video{folder_number}_images/feature_dict{n//100}.pth')  # the target dir for saving features.
        tensor_dict.clear()
        print(n)
    if n == count:
        if n % 100 !=0:
            torch.save(tensor_dict, f'target_dir/Wu2017_features/video{folder_number}_images/feature_dict{(n//100)+1}.pth')
        tensor_dict.clear()


if __name__ == "__main__":
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    # the source dir for storing saliency maps
    subdirectories = ['source_dir/Wu2017_saliencyMap/video{}_images'.format(i) for i in range(1, 10)]
    processeachsub(subdirectories, start_folder=1)