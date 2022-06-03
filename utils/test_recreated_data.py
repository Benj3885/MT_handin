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


def get_dataloader(data_paths, size=256, good_per_defect=1., batch_size=32, max_defect_imgs=inf):

    transform = transforms.Compose(
        [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        ]
    )

    training_data = ClassificationDataset(data_paths, transform=transform, good_per_defect=good_per_defect, max_defect_imgs=max_defect_imgs)
    return torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=4)


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


def create_ROC(ml_client, ml_run, labels_all, preds_all, pre_ROC="", pre_AUC=""):
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
    ml_client.log_metric(ml_run.info.run_id, pre_AUC + "AUC", AUC)
    plt.close(fig)

    return AUC


def handle_hard_samples(samples, cat, save_location, device, generator):
    generator.to(device)
    for sample in (pbar := tqdm(samples)):
        pbar.set_description((f"creating {cat[:-1]} new samples"))
        img_score = sample[0]
        img_path = sample[1]
        img_in = Image.open(img_path)
        img_out = transform_sample(img_in)
        fn = os.path.basename(img_path)
        fn_new = f"{cat}{img_score}_" + fn
        ml_client.log_image(ml_run.info.run_id, img_out, fn_new)

        new_samples = generator.augment(img_in, 1, 
                                        bot_row_ratio=data_ops['row_ratio'], 
                                        row_interval=data_ops['row_interval'], 
                                        col_interval=data_ops['col_interval']
                                        )

        for sample in new_samples:
            sample.save(save_location + fn)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    val_dataloader = get_dataloader(data_ops['validation'], good_per_defect=data_ops['good_per_defect'])
    test_dataloader = get_dataloader(data_ops['test'], good_per_defect=data_ops['good_per_defect'])

    # Mlflow setup
    ml_client = mlflow.tracking.MlflowClient(tracking_uri="file:" + data_ops['tracking_uri'])
    exp_number = str(len(ml_client.list_experiments()))
    exp_id = "Exp" + exp_number
    ml_client.create_experiment(exp_id)

    log_run = ml_client.create_run(experiment_id=exp_number)
    ml_client.log_artifact(log_run.info.run_id, "test_recreated_data.py")
    ml_client.log_artifact(log_run.info.run_id, "settings/data.yaml")
    ml_client.log_artifact(log_run.info.run_id, "settings/model_settings.yaml")

    total_test_AUC = 0.
    for iteration in range(model_ops['iterations']):
        ml_run = ml_client.create_run(experiment_id=exp_number)
        classifier = get_classifier(size_option=model_ops['size_option'], num_classes=model_ops['num_classes'], in_chans=model_ops['in_chans'], device=device)
        optimizer = torch.optim.SGD(classifier.parameters(), lr=model_ops['lr'], momentum=model_ops['momentum'])

        # Dataloaders
        train_dataloader = get_dataloader(train_data_loader_paths, good_per_defect=data_ops['good_per_defect'], max_defect_imgs=data_ops['max_defect_imgs'])

        for epoch in range(model_ops['epochs']):  # loop over the dataset multiple times

            train_running_loss = 0.0
            train_total_images = 0
            for i, data in enumerate(pbar := tqdm(train_dataloader)):
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

                pbar.set_description((f"epoch: {epoch}; loss: {loss.item():.5f}; average loss: {train_avg_loss:.5f}"))

            # Internal validation
            with torch.no_grad():
                val_correct = 0
                val_total_images = 0
                val_running_loss = 0.0

                labels_all = []
                preds_all = []
                for data in (pbar := tqdm(val_dataloader)):
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

                AUC = create_ROC(ml_client, ml_run, labels_all, preds_all, pre_ROC="val_"+str(epoch)+ "_", pre_AUC="val_")
                ml_client.log_metric(ml_run.info.run_id, "train_loss", train_avg_loss)
                ml_client.log_metric(ml_run.info.run_id, "val_loss", avg_val_loss)
                ml_client.log_metric(ml_run.info.run_id, "val_acc", val_acc)


        with torch.no_grad():
            # Test
            labels_all = []
            preds_all = []

            test_correct = 0
            test_total_images = 0
            test_running_loss = 0.0
            for data in (pbar := tqdm(test_dataloader)):
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



            AUC = create_ROC(ml_client, ml_run, labels_all, preds_all, pre_ROC="test_", pre_AUC="test_")
            total_test_AUC += AUC
            ml_client.log_metric(ml_run.info.run_id, "test_loss", avg_test_loss)
            ml_client.log_metric(ml_run.info.run_id, "test_acc", test_acc)
            ml_client.log_metric(log_run.info.run_id, "avg_AUC", total_test_AUC / (iteration+1))
