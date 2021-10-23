import argparse
from datasets.dataset_synapse import JigsawTransformation
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import parameter
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import matplotlib.pyplot as plt
from networks.vit_seg_modeling_resnet_skip import jigsaw_softmax

def trainer_synapse_semisupervised(args, model_semisupervised, model_main, snapshot_path):
    
    # Setup information logging
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(snapshot_path + '/log')

    # Set up training parameters
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
   
    # Set up dataset
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator


    jigaw_transformation = JigsawTransformation(model_semisupervised.K, model_semisupervised.Q, 225)
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                    [
                                        RandomGenerator(output_size=[args.img_size, args.img_size]),
                                        jigaw_transformation
                                    ]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    # Set up training model and losses
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if args.n_gpu > 1:
        model_main = nn.DataParallel(model_main)
        model_semisupervised = nn.DataParallel(model_semisupervised)

    model_main.train()
    model_semisupervised.train()

    # Set up training variables
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer_overall = optim.SGD(list(model_main.parameters()) + list(model_semisupervised.parameters()), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer_jigsaw = optim.SGD(model_semisupervised.parameters(), lr=base_lr, momentum=0.9, weight_decay = 5e-4)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    # max_iterations = args.max_iterations

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    if (args.continue_from_epoch > 0):
        logging.info(f"Continue training from epoch {args.continue_from_epoch}")
    else:
        logging.info("Starting from epoch 0")

    # Start training
    best_performance = 0.0
    iterator = tqdm(range(args.continue_from_epoch, max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            unlabelled = True

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device=device, dtype=torch.float), label_batch.to(device=device, dtype=torch.long)
            

            image_batch_jigsaw, label_batch_jigsaw = sampled_batch['jigsaw_images'], sampled_batch['jigsaw_labels']
            image_batch_jigsaw, label_batch_jigsaw = image_batch_jigsaw.to(device=device, dtype=torch.float), label_batch_jigsaw.to(device=device, dtype=torch.long)

            # Image transformation checking
            # for i in range(len(sampled_batch)):
            #     image_batch_nocuda = image_batch.to(device="cpu")
            #     jig_batch_nocuda = image_batch_jigsaw.to(device="cpu")

            #     img = image_batch_nocuda[i].squeeze(0)
            #     jig = jig_batch_nocuda[i][0].squeeze(0)
            #     fig, (ax1, ax2) = plt.subplots(2)
            #     ax1.imshow(img)
            #     ax2.imshow(jig)

            #     plt.show()
            #     input()

            image_batch_jigsaw = image_batch_jigsaw.transpose(0, 1) # B, Q, ... -> Q, B，...
            label_batch_jigsaw = label_batch_jigsaw.transpose(0, 1)

            # # Jigsaw classification task for Q different transformations
            jigsaw_resnet_outputs = []
            jigsaw_features = []
            loss_jigsaw_list = []
            for q in range(model_semisupervised.Q):
                jigsaw_classificaiton_outputs, resnet_outputs, features = model_semisupervised(image_batch_jigsaw[q])
                jigsaw_resnet_outputs.append(resnet_outputs)
                jigsaw_features.append(features)
                # print(f"\n\n\n jigsaw output size: {jigsaw_classificaiton_outputs.size()} label batch size: {label_batch_jigsaw[q].size()}\n\n\n")
                loss_jigsaw = ce_loss(jigsaw_classificaiton_outputs, label_batch_jigsaw[q].flatten())
                loss_jigsaw_list.append(loss_jigsaw.item())

                optimizer_overall.zero_grad()
                loss_jigsaw.backward()
                optimizer_overall.step()
            
            # Original data pass through
            _, resnet_outputs, features = model_semisupervised(image_batch)
            outputs_main = model_main(resnet_outputs, features)

            # Unlabelled data
            if unlabelled:
                pseudo_label_batch = np.zeros(image_batch_jigsaw[0].size()) 
                for q in range(model_semisupervised.Q):
                    outputs_main = model_main(jigsaw_resnet_outputs[q], jigsaw_features[q])
                    restored_outputs = jigaw_transformation.restore(outputs_main, label_batch_jigsaw[q])
                    for b in range(len(outputs_main)):
                        pseudo_label_batch[b] += restored_outputs[b] * jigsaw_softmax(q, loss_jigsaw_list)
                loss_ce_main = ce_loss(outputs_main, pseudo_label_batch)
                loss_dice_main = dice_loss(outputs_main, pseudo_label_batch, softmax=True)
                loss_main = 0.5 * loss_ce_main + 0.5 * loss_dice_main
            else:
                loss_ce_main = ce_loss(outputs_main, label_batch)
                loss_dice_main = dice_loss(outputs_main, label_batch, softmax=True)
                loss_main = 0.5 * loss_ce_main + 0.5 * loss_dice_main
            
            optimizer_overall.zero_grad()
            loss_main.backward()
            optimizer_overall.step()
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer_overall.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss_main', loss_main, iter_num)
            writer.add_scalar('info/loss_ce_main', loss_ce_main, iter_num)

            logging.info(f'iteration: {iter_num:d} loss_main: {loss_main.item():f},',
                f'loss_ce_main: {loss_ce_main.item():f} loss jigsaw total: {sum(loss_jigsaw_list):.4f} unlabelled: {unlabelled}')

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs_main = torch.argmax(torch.softmax(outputs_main, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_main[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path_main = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '_main.pth')
            save_mode_path_semi = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '_semi.pth')
            torch.save(model_main.state_dict(), save_mode_path_main)
            torch.save(model_semisupervised.state_dict(), save_mode_path_semi)
            logging.info("save model to {}".format(save_mode_path_main))

        if epoch_num >= max_epoch - 1:
            save_mode_path_main = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '_main.pth')
            save_mode_path_semi = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '_semi.pth')
            torch.save(model_main.state_dict(), save_mode_path_main)
            torch.save(model_semisupervised.state_dict(), save_mode_path_semi)
            logging.info("save model to {}".format(save_mode_path_main))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"