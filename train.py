import argparse
import collections
import os
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, \
    AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet.eval import Evaluation
    
from torch.utils.data import DataLoader

from torchviz import make_dot

def set_seed(seed_value):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print('[*] (GPU Available) seed is successfully set to {}'.format(seed_value))
    else:
        print('[*] seed is successfully set to {}'.format(seed_value))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='./data')
    parser.add_argument('--output_path', help='Path to output directory to save checkpoints', default='./output')
    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=72)
    parser.add_argument('--seed', help='Set the random seed for reproducibility', type=int, default=3407)
    parser.add_argument('--learning_rate', help='Set the learning rate for the optimizer', type=float, default=1e-4)
    parser.add_argument('--draw_model', help='Draw the model or not, if draw then no train', type=bool, default=False)

    parser = parser.parse_args(args)

    # Set the random seed for reproducibility
    set_seed(parser.seed)

    if not os.path.exists(parser.output_path):
        os.mkdir(parser.output_path)

    if parser.coco_path is None:
        raise ValueError('Must provide --coco_path when training on COCO.')

    dataset_train = CocoDataset(parser.coco_path, set_name='train',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(parser.coco_path, set_name='val',
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=2, collate_fn=collater, batch_sampler=sampler)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=parser.learning_rate)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[48, 64])

    loss_hist = collections.deque(maxlen=500)
    epoch_loss_list = []

    # Draw the structure of model
    if parser.draw_model:
        # Get a batch of training data
        # Get a batch of training data
        data = next(iter(dataloader_train))
        inputs = data['img']
        labels = data['annot']

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Run the model on the inputs
        outputs = retinanet([inputs, labels])

        # Generate the graph
        dot = make_dot(outputs, params=dict(retinanet.named_parameters()))

        # Save the graph
        dot.format = 'png'
        dot.render(filename='retinanet')
        return

    print('Num training images: {}'.format(len(dataset_train)))
    for epoch_num in range(parser.epochs):
        
        retinanet.training = True
        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in tqdm(enumerate(dataloader_train)):
            
            ###################################################################
            # TODO: Please fill the codes here to zero optimizer gradients
            ##################################################################
            optimizer.zero_grad()

            ##################################################################

            if torch.cuda.is_available():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            
            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            ###################################################################
            # TODO: Please fill the codes here to complete the gradient backward
            ##################################################################
            loss.backward()

            ##################################################################

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            ###################################################################
            # TODO: Please fill the codes here to optimize parameters
            ##################################################################
            optimizer.step()

            ##################################################################

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            if iter_num % 100 == 0:
                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss

        scheduler.step()

        epoch_loss_list.append(np.mean(epoch_loss))

        if (epoch_num + 1) % 10 == 0 or epoch_num + 1 == parser.epochs:
            print('epoch_loss_list:')
            print(epoch_loss_list)
            
            print('Evaluating dataset')
            retinanet.eval()
            retinanet.training = False
            eval = Evaluation()
            eval.evaluate(dataset_val, retinanet)

            torch.save(retinanet, os.path.join(parser.output_path, 'retinanet_epoch{}.pt'.format(epoch_num + 1)))

    print('epoch_loss_list:')
    print(epoch_loss_list)
    torch.save(retinanet, os.path.join(parser.output_path, 'model_final.pt'))


if __name__ == '__main__':
    main()
