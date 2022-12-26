import os
import csv
import math
import torch
import config
import pickle
import numpy as np
from utils.lid_utils import LID
from utils.utils import (
    network_initialization, set_seed, get_dataloader
    )
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV

def main(args):
    # check necessary files
    assert os.path.isfile(f'baselineCNN/best_model/{args.model}/{args.dataset}.pt'), \
        'model file not found... must first train model'
    assert os.path.isfile(f'adv_examples/{args.model}/{args.dataset}/{args.attack_save_path}.pt'), \
        'adversarial sample file not found... must first craft adversarial samples'

    # set save path and create the path if not exist
    lid_results_path = f'results/lid'
    os.makedirs(f'{lid_results_path}/lid_features/{args.attack_save_path}/{args.model}', exist_ok=True)
    os.makedirs(f'{lid_results_path}/{args.attack_save_path}', exist_ok=True)

    #------generate characteristics
    print('Loading the data and model...')
    model = network_initialization(args)
    model.load_state_dict(
        torch.load(f'baselineCNN/best_model/{args.model}/{args.dataset}.pt')["model_state_dict"],
    )
    model.eval()

    # Load the dataset
    _, _, data_loader = get_dataloader(args)

    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    normal_imgs = None
    idx_correct = None
    test_labels = None
    cum = 0
    for step, (imgs, labels) in enumerate(data_loader):
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        idx_correct_batch = torch.where(predicted.eq(labels))[0].to(args.device)
        if normal_imgs is None:
            normal_imgs = imgs[idx_correct_batch]
            test_labels = labels[idx_correct_batch]
            idx_correct = idx_correct_batch
        normal_imgs = torch.cat((normal_imgs, imgs[idx_correct_batch]))
        test_labels = torch.cat((test_labels, labels[idx_correct_batch]))
        idx_correct = torch.cat((idx_correct, idx_correct_batch+cum))
        cum += labels.size(0)
    print("Number of correctly predict images: %s" % (len(idx_correct)))

    # Check attack type, select adversarial and noisy samples accordingly
    print('Loading noisy and adversarial samples...')
    # Load adversarial samples
    adv_imgs, _ = torch.load(f'./adv_examples/{args.model}/{args.dataset}/{args.attack_save_path}.pt')
    adv_imgs = adv_imgs[idx_correct].to(args.device)

    # as there are some parameters to tune for noisy example, so put the generation
    # step here instead of the adversarial step which can take many hours
    print('Crafting %s noisy samples. ' % args.dataset)
    noise = torch.normal(0, args.eps, size=normal_imgs.size(), device=normal_imgs.device)
    noisy_imgs = normal_imgs + noise

    # extract local intrinsic dimensionality --- load if it existed
    lid_adv_path = f'{lid_results_path}/lid_features/{args.attack_save_path}/{args.model}/{args.dataset}_lid_adv.pt'
    lid_noisy_path = f'{lid_results_path}/lid_features/{args.attack_save_path}/{args.model}/{args.dataset}_lid_noisy.pt'
    lid_normal_path = f'{lid_results_path}/lid_features/{args.attack_save_path}/{args.model}/{args.dataset}_lid_normal.pt'
    # if os.path.isfile(lid_adv_path) & os.path.isfile(lid_noisy_path) & os.path.isfile(lid_normal_path):
    #     with open(lid_normal_path, 'rb') as f:
    #         lid_normal = pickle.load(f)
    #     with open(lid_noisy_path, 'rb') as f:
    #         lid_noisy = pickle.load(f)
    #     with open(lid_adv_path, 'rb') as f:
    #         lid_adv = pickle.load(f)
    # else:
        # # LID inheritance
        # lid = LID(model, normal_imgs, adv_imgs, noisy_imgs, args.batch_size, args.lid_k)
        # lid_normal, lid_noisy, lid_adv = lid.get_lids_random_batch()
        # with open(lid_adv_path, 'wb') as f:
        #     pickle.dump(lid_adv, f)
        # with open(lid_noisy_path, 'wb') as f:
        #     pickle.dump(lid_noisy, f)
        # with open(lid_normal_path, 'wb') as f:
        #     pickle.dump(lid_normal, f)
        # print("lids_normal:", lid_normal.shape)
        # print("lids_noisy:", lid_noisy.shape)
        # print("lids_adv:", lid_adv.shape)

    #################################################################################
    lid = LID(model, normal_imgs, adv_imgs, noisy_imgs, args.batch_size, args.lid_k)
    lid_normal, lid_noisy, lid_adv = lid.get_lids_random_batch()
    with open(lid_adv_path, 'wb') as f:
        pickle.dump(lid_adv, f)
    with open(lid_noisy_path, 'wb') as f:
        pickle.dump(lid_noisy, f)
    with open(lid_normal_path, 'wb') as f:
        pickle.dump(lid_normal, f)
    print("lids_normal:", lid_normal.shape)
    print("lids_noisy:", lid_noisy.shape)
    print("lids_adv:", lid_adv.shape)


    # Split the dataset into train and test
    # (Split same as Official LID code)
    split_idx = int(len(lid_normal)*0.8)
    x_train = np.r_[(lid_normal[:split_idx], lid_noisy[:split_idx], lid_adv[:split_idx])]
    y_train = np.r_[(np.zeros(len(lid_normal[:split_idx])), np.zeros(len(lid_noisy[:split_idx])), np.ones(len(lid_adv[:split_idx])))]

    # Test
    lid_normal_test = lid_normal[split_idx:]
    lid_adv_test = lid_adv[split_idx:]
    x_test = np.r_[(lid_normal_test, lid_adv_test)]
    y_test = np.r_[(np.zeros(len(lid_normal_test)), np.ones(len(lid_adv_test)))]

    # Build detector
    print("LR Detector on [dataset: %s, train_attack: %s, target_attack: %s] with:" % (args.dataset, args.attack_save_path, args.target_attack))
    lr = LogisticRegressionCV(n_jobs=7, max_iter=3000, solver='lbfgs').fit(x_train, y_train)

    # Check if attack is success or not
    adv_test_imgs = adv_imgs[split_idx:]
    test_labels = test_labels[split_idx:]

    n_iter = int(math.ceil(len(adv_test_imgs)/args.batch_size))
    idx_attack_success = torch.tensor([False]*len(test_labels), device=adv_imgs.device)
    cum = 0
    with torch.no_grad():
        for step in range(n_iter):
            # make mini-batch
            start = args.batch_size*step
            end = np.minimum(args.batch_size*(step+1), len(adv_test_imgs))
            batch_input = adv_test_imgs[start:end]
            batch_label = test_labels[start:end]
            # get distances
            out = model(batch_input)
            _, predicted = torch.max(out, 1)
            # save examples which is misclassified from classifier (Save only adversarial examples)
            batch_idx_attack_success = torch.where(~predicted.eq(batch_label))[0].to(args.device)
            idx_attack_success[batch_idx_attack_success+cum] = True
            cum += (end-start)

    idx_attack_success = idx_attack_success.cpu().numpy()
    X_success = np.r_[(lid_normal_test[idx_attack_success], lid_adv_test[idx_attack_success])]
    y_success = np.r_[(np.zeros(len(lid_normal_test[idx_attack_success])), np.ones(len(lid_adv_test[idx_attack_success])))]

    X_fail = np.r_[(lid_normal_test[~idx_attack_success], lid_adv_test[~idx_attack_success])]
    y_fail = np.r_[(torch.zeros(len(lid_normal_test[~idx_attack_success])), torch.ones(len(lid_adv_test[~idx_attack_success])))]

    # save results
    results_all = []
    for X, y, t in [[x_test, y_test, 'all'], [X_success, y_success, 'success'], [X_fail, y_fail, 'fail']]:
        try:
            Y_pred_score = lr.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, Y_pred_score)
            auroc = auc(fpr, tpr)
            curr_result = {'type':t, 'nsamples': len(X), 'tpr': list(tpr), 'fpr': list(fpr), 'auc': auroc}
        except:
            curr_result = {'type':t, 'nsamples': 0, 'tpr': None, 'fpr': None, 'auc': None}
        results_all.append(curr_result)

    # save a s csv file
    with open(f'{lid_results_path}/{args.attack_save_path}/[{args.dataset}]_{args.model}.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=['type', 'nsamples', 'tpr', 'fpr', 'auc']
            )
        writer.writeheader()
        for row in results_all:
            writer.writerow(row)
    print('Done!')

if __name__ == "__main__":
    args = config.get_config()
    set_seed(args.seed)
    main(args)
