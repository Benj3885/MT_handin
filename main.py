import torch
from generators.generators import vqvae_generator
from utils.dataset import ClassificationDataset
from torchvision import transforms
from classifiers.ConvNeXt.models.convnext import ConvNeXt
from tqdm import tqdm
from cmath import inf
import mlflow
import yaml
from sklearn import metrics
from matplotlib import pyplot as plt
from PIL import Image
import os
import random
import threading
import numpy as np


def get_dataloader(data_paths, size=256, good_per_defect=1., batch_size=32, max_defect_imgs=inf, shuffle=True, in_chans=1):

    transform = transforms.Compose(
        [
        transforms.Grayscale(num_output_channels=in_chans),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        ]
    )

    training_data = ClassificationDataset(data_paths, transform=transform, good_per_defect=good_per_defect, max_defect_imgs=max_defect_imgs)
    return torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)


def get_classifier(size_option='tiny', num_classes=2, in_chans=1, device='cuda'):
    if size_option == 'tiny':
        return ConvNeXt(num_classes=num_classes, in_chans=in_chans, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(device)
    elif size_option == 'small':
        return ConvNeXt(num_classes=num_classes, in_chans=in_chans, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]).to(device)
    elif size_option == 'base':
        return ConvNeXt(num_classes=num_classes, in_chans=in_chans, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]).to(device)
    elif size_option == 'large':
        return ConvNeXt(num_classes=num_classes, in_chans=in_chans, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]).to(device)
    elif size_option == 'xlarge':
        return ConvNeXt(num_classes=num_classes, in_chans=in_chans, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048]).to(device)


def create_ROC(ml_client, ml_run, labels_all, preds_all, pre_ROC=""):
    fpr, tpr, _ = metrics.roc_curve(labels_all, preds_all)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, '-')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.grid()
    AUC = metrics.auc(fpr, tpr)
    ax.set_title(f'AUC: {round(AUC, 4)}')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.05, 1.05)
    ml_client.log_figure(ml_run.info.run_id, fig, pre_ROC + "ROC.png")
    plt.close(fig)

    return AUC


def handle_hard_samples(samples, save_location, device, generator, iteration):
    generator.to(device)
    for sample in (pbar := tqdm(samples, leave=False)):
        pbar.set_description((f"Creating new samples on {device}"))
        img_score = sample[0]
        img_label = sample[1]
        img_path = sample[2]
        img_in = Image.open(img_path)
        img_out = transform_sample(img_in)
        fn = os.path.basename(img_path)
        cat = ("good" if img_label == 0 else "defect")
        fn_new = f"{cat}_{img_score}_" + fn
        ml_client.log_image(ml_run.info.run_id, img_out, fn_new)

        new_samples = generator.augment(img_in, 1, 
                                        bot_row_ratio=data_ops['row_ratio'], 
                                        row_interval=data_ops['row_interval'], 
                                        col_interval=data_ops['col_interval']
                                        )

        for sample in new_samples:
            sample.save(save_location + f"{cat}/" + str(iteration) + "_" + fn)


def get_confidences(classifier, dataloader, device, sample_labels):
    with torch.no_grad():
        img_conf = [] 
        
        for data in (pbar := tqdm(dataloader)):
            pbar.set_description((f"Getting confidences"))

            inputs, labels, paths = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = classifier(inputs)

            # Hard samples
            for i in range(len(labels)):
                label = labels[i]
                if label not in sample_labels:
                    continue
                score = outputs[i, label]
                img_conf.append((score, label, paths[i]))
                
        return sorted(img_conf, key=lambda tup: tup[2])



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_ops = yaml.load(open('settings/data.yaml'), Loader=yaml.SafeLoader)
    model_ops = yaml.load(open('settings/model_settings.yaml'), Loader=yaml.SafeLoader)

    # Seeding for reproducibility
    random.seed(model_ops['random_seed'])
    torch.manual_seed(model_ops['random_seed'])

    # Transform for logging the samples with high loss. Will convert to model input size
    transform_sample = transforms.Compose(
        [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((model_ops['size'], model_ops['size']))
        ]
    )

    # Classification
    criterion = torch.nn.CrossEntropyLoss()
    train_data_loader_paths = data_ops['train']
    val_dataloader = get_dataloader(data_ops['validation'], good_per_defect=data_ops['good_per_defect'], shuffle=True)
    test_dataloader = get_dataloader(data_ops['test'], good_per_defect=data_ops['good_per_defect'], shuffle=False)

    # Mlflow setup
    ml_client = mlflow.tracking.MlflowClient(tracking_uri="file:" + data_ops['tracking_uri'])
    exp_number = str(len(ml_client.list_experiments()))
    exp_id = "Exp" + exp_number
    ml_client.create_experiment(exp_id)

    # Setting up directory for new samples
    new_samples_path = data_ops['new_samples_path'] + exp_id + "/"
    train_data_loader_paths[0].append(new_samples_path + "good")
    train_data_loader_paths[1].append(new_samples_path + "defect")
    os.makedirs(new_samples_path)
    os.makedirs(new_samples_path + "defect")
    os.makedirs(new_samples_path + "good")

    log_run = ml_client.create_run(experiment_id=exp_number)
    ml_client.log_artifact(log_run.info.run_id, "main.py")
    ml_client.log_artifact(log_run.info.run_id, "settings/data.yaml")
    ml_client.log_artifact(log_run.info.run_id, "settings/model_settings.yaml")

    for iteration in (iter_pbar := tqdm(range(model_ops['iterations']))):
        if iteration == 0:
            iter_pbar.set_description((f"Iteration"))

        ml_run = ml_client.create_run(experiment_id=exp_number)

        train_dataloader = get_dataloader(train_data_loader_paths, good_per_defect=data_ops['good_per_defect'])

        accum_model_confs = None

        test_AUCs = []
        all_cnv_epochs = []

        for m in (models_pbar := tqdm(range(model_ops['models_per_iteration']), leave=False)):
            models_pbar.set_description((f"Model"))
            classifier = get_classifier(size_option=model_ops['size_option'], num_classes=model_ops['num_classes'], in_chans=model_ops['in_chans'], device=device)
            optimizer = torch.optim.SGD(classifier.parameters(), lr=model_ops['lr'], momentum=model_ops['momentum'])

            lowest_val_loss = inf
            ckpt = None
            cnv_epoch = None

            for epoch in (epoch_pbar := tqdm(range(model_ops['epochs']), leave=False)):  # loop over the dataset multiple times

                train_running_loss = 0.0
                train_total_images = 0
                for i, data in enumerate(pbar := tqdm(train_dataloader, leave=False)):
                    optimizer.zero_grad()

                    inputs, labels, _ = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    # forward + backward + optimize
                    outputs = classifier(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Statistics and updates
                    train_total_images += len(labels)
                    train_running_loss += loss.item()
                    train_avg_loss = train_running_loss / train_total_images

                    pbar.set_description((f"loss: {loss.item():.5f}; average loss: {train_avg_loss:.5f}"))

                # Internal validation
                with torch.no_grad():
                    val_correct = 0
                    val_total_images = 0
                    val_running_loss = 0.0

                    labels_all = []
                    preds_all = []
                    for data in (pbar := tqdm(val_dataloader, leave=False)):
                        inputs, labels, _ = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        outputs = classifier(inputs)

                        # Loss statistics
                        val_loss = criterion(outputs, labels)
                        val_running_loss += val_loss.item()
                        val_total_images += len(labels)
                        avg_val_loss = val_running_loss / val_total_images

                        # Accuracy statistics
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_acc = val_correct / val_total_images

                        # Feedback
                        pbar.set_description((f"average val loss: {avg_val_loss:.5f}; "
                                            f"val acc: {val_acc:.5f}"))

                        # Information for ROC curve
                        for l in labels.cpu().numpy():
                            labels_all.append(l)
                        for o in outputs.cpu().numpy():
                            preds_all.append(o[1])

                    AUC = create_ROC(ml_client, ml_run, labels_all, preds_all, pre_ROC=f"val_{m}_{epoch}_")
                    ml_client.log_metric(ml_run.info.run_id, f"{m}_train_loss", train_avg_loss)
                    ml_client.log_metric(ml_run.info.run_id, f"{m}_val_loss", avg_val_loss)
                    ml_client.log_metric(ml_run.info.run_id, f"{m}_val_acc", val_acc)

                    if avg_val_loss < lowest_val_loss:
                        lowest_val_loss = avg_val_loss
                        ckpt = classifier.state_dict()
                        cnv_epoch = epoch


            with torch.no_grad():
                classifier.load_state_dict(ckpt)

                # Test
                labels_all = []
                preds_all = []

                test_correct = 0
                test_total_images = 0
                test_running_loss = 0.0
                for data in (pbar := tqdm(test_dataloader, leave=False)):
                    inputs, labels, _ = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = classifier(inputs)

                    # Loss statistics
                    test_loss = criterion(outputs, labels)
                    test_running_loss += test_loss.item()
                    test_total_images += len(labels)
                    avg_test_loss = test_running_loss / test_total_images

                    # Accuracy statistics
                    _, predicted = torch.max(outputs.data, 1)
                    test_correct += (predicted == labels).sum().item()
                    test_acc = test_correct / test_total_images

                    pbar.set_description((f"average test loss: {avg_test_loss:.5f}; "
                                        f"test acc: {test_acc:.5f}"))

                    # ROC
                    for l in labels.cpu().numpy():
                        labels_all.append(l)

                    for o in outputs.cpu().numpy():
                        preds_all.append(o[1])

                AUC = create_ROC(ml_client, ml_run, labels_all, preds_all, pre_ROC="test_")
                ml_client.log_metric(ml_run.info.run_id, "test_loss", avg_test_loss)
                ml_client.log_metric(ml_run.info.run_id, "test_acc", test_acc)
                ml_client.log_metric(ml_run.info.run_id, "AUC", AUC)
                ml_client.log_metric(ml_run.info.run_id, "cnv_epoch", cnv_epoch)
                test_AUCs.append(AUC)
                all_cnv_epochs.append(cnv_epoch)

                iter_pbar.set_description((f"Avg. AUC: {np.mean(test_AUCs)}  Iteration"))

        if accum_model_confs == None:
            accum_model_confs = get_confidences(classifier, train_dataloader, device, data_ops['new_labels'])
        else:
            confs_for_model = get_confidences(classifier, train_dataloader, device, data_ops["new_labels"])
            for i in range(len(accum_model_confs)):
                accum_model_confs[i][0] = accum_model_confs[i][0] + confs_for_model[i][0]


        ml_client.log_metric(log_run.info.run_id, f"avg_AUC", np.mean(test_AUCs))
        ml_client.log_metric(log_run.info.run_id, f"std_AUC", np.std(test_AUCs))
        ml_client.log_metric(log_run.info.run_id, f"avg_cnv_epoch", np.mean(all_cnv_epochs))
        ml_client.log_metric(log_run.info.run_id, f"std_cnv_epoch", np.std(all_cnv_epochs))

        if data_ops['augment_samples']:
            #Save hard samples and create new ones
            accum_model_confs = sorted(accum_model_confs, key=lambda tup: tup[0])
            for conf in accum_model_confs:
                conf = (conf[0] / model_ops['models_per_iteration'], conf[1], conf[2])

            threads = []
            for i in range(model_ops['n_gpus']):
                start = i*data_ops['n_hard_samples']//model_ops['n_gpus']
                stop = start + data_ops['n_hard_samples']//model_ops['n_gpus']
                samples = accum_model_confs[start:stop]

                
                generator = vqvae_generator(model_ops['vqvae_ckpt'], model_ops['bot_ckpt'], model_ops['size'])
                thread = threading.Thread(target=handle_hard_samples, args=(samples, new_samples_path, f'cuda:' + str(i), generator, iteration,))
                thread.start()
                threads.append(thread)

                if len(threads) == model_ops['n_gpus'] :
                    for thread in threads:
                        thread.join()
