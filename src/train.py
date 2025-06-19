import argparse
import os
import time

import numpy as np
import torch
import tqdm
from code.generate_plots import save_results_to_csv
from losses import CombinedLoss, FocalFrequencyLoss
from matplotlib import pyplot as plt
from networks import get_model
from utils import per_class_dice

from data import get_data


def argument_parser():
    parser = argparse.ArgumentParser()

    # Optimization hyperparameters
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument(
        "--num_iterations", default=5, type=int, help="Number of Epochs"
    )
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--n_classes", default=9, type=int)
    parser.add_argument("--ffc_lambda", default=0, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)

    ## Non DP training ##

    # Dataset options
    parser.add_argument("--dataset", default="Pneumothorax", choices=["Pneumothorax"])
    parser.add_argument("--image_size", default="224", type=int)
    parser.add_argument(
        "--image_dir",
        default="data",
        choices=["data"],
    )
    parser.add_argument(
        "--model_name", default="NestedUNet", choices=["unet", "NestedUNet", "ConvNet"]
    )

    # Network options
    parser.add_argument("--g_ratio", default=0.5, type=float)

    # Other options
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--in_channels", default=1, type=int)

    return parser


def colored_text(st):
    return "\033[91m" + st + "\033[0m"


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def eval(
    val_loader,
    criterion,
    model,
    n_classes,
    dataset,
    algorithm,
    location,
    dice_s=True,
    device="cpu",
    im_save=True,
):
    model.eval()
    loss = 0
    counter = 0
    dice = 0
    correct_pixels = 0
    total_pixels = 0

    dice_all = np.zeros(n_classes)

    for img, label in tqdm.tqdm(val_loader):
        img = img.to(device)
        label = label.to(device)
        label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes)

        pred = model(img)
        max_val, idx = torch.max(pred, 1)
        pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)

        if dice_s:
            d1, d2 = per_class_dice(pred_oh, label_oh, n_classes)
            dice += d1
            dice_all += d2

        loss += criterion(pred, label.squeeze(1), device=device).item()

        # Calculate accuracy
        correct_pixels += (idx == label.squeeze(1)).sum().item()
        total_pixels += torch.numel(label.squeeze(1))

        if im_save:
            # Save the predicted segmentation and the ground truth segmentation
            name = f"{algorithm}-{counter}"
            fig, ax = plt.subplots(1, 2)
            fig.suptitle(name, fontsize=10)

            ax[0].imshow(label.squeeze().detach().cpu().contiguous().numpy(), cmap="gray")
            ax[0].set_title(f"Ground Truth")
            ax[1].imshow(idx.cpu().squeeze().numpy(), cmap="gray")
            ax[1].set_title(f"Prediction")
            fig.subplots_adjust(top=0.85)

            dir_path = f"results/{algorithm}/{location}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            plt.savefig(f"results/{algorithm}/{location}/{name}")
            plt.close(fig)

        counter += 1

    loss = loss / counter
    dice = dice / counter
    dice_all = dice_all / counter
    accuracy = correct_pixels / total_pixels
    print(
        "Validation loss: ",
        loss,
        " Mean Dice: ",
        dice.item(),
        "Dice All:",
        dice_all,
        "Accuracy: ",
        accuracy,
    )
    return dice, loss, accuracy


def train(args):
    # Optimization hyperparameters
    batch_size = args.batch_size
    iterations = args.num_iterations
    learning_rate = args.learning_rate
    n_classes = args.n_classes

    # Dataset options
    dataset = args.dataset
    img_size = args.image_size
    data_path = args.image_dir
    model_name = args.model_name

    # Network options
    ratio = args.g_ratio

    # Other options
    device = args.device
    training_losses = []
    validation_losses = []
    validation_dice_scores = []
    validation_accuracies = []
    criterion_seg = CombinedLoss()
    criterion_ffc = FocalFrequencyLoss()

    algorithm = "Non-DP"
    save_name = f"results/{algorithm}/{model_name}.pt"
    file_name = f"results/{algorithm}/{model_name}.csv"

    max_dice = 0
    # best_test_dice = 0
    best_iter = 0

    model = get_model(model_name, ratio=ratio, num_classes=n_classes).to(device)
    model.train()

    optimizer = torch.optim.SGD(
        list(model.parameters()), lr=learning_rate, weight_decay=args.weight_decay
    )

    train_loader, val_loader, test_loader, _, _, _ = get_data(
        data_path, img_size, batch_size
    )

    start_time = time.time()
    iteration_train_times = []
    for t in range(iterations):
        start_epoch_time = time.time()
        total_loss = 0
        total_samples = 0

        for img, label in tqdm.tqdm(train_loader):
            img = img.to(device)
            label = label.to(device)
            label_oh = torch.nn.functional.one_hot(
                label, num_classes=n_classes
            ).squeeze()

            pred = model(img)
            max_val, idx = torch.max(pred, 1)
            pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
            pred_oh = pred_oh.permute(0, 3, 1, 2)
            label_oh = label_oh.permute(0, 3, 1, 2)
            loss = criterion_seg(
                pred, label.squeeze(1), device=device
            ) + args.ffc_lambda * criterion_ffc(pred_oh, label_oh)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = loss.item() + img.size(0)
            total_samples += img.size(0)

        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time
        iteration_train_times.append(epoch_time)
        average_loss = total_loss / total_samples
        training_losses.append(average_loss)

        print(
            f"\tTrain Epoch: [{t + 1}/{iterations}] \t"
            f"Train Epoch Average Loss: {np.mean(average_loss):.6f} \t"
            f"Train Epoch Time: {epoch_time:.2f} \t"
        )

        if t % 10 == 0 or t > 45:
            dice, validation_loss, accuracy = eval(
                val_loader,
                criterion_seg,
                model,
                dice_s=True,
                n_classes=n_classes,
                dataset=dataset,
                algorithm=algorithm,
                location=model_name,
            )
            validation_losses.append(validation_loss)
            validation_dice_scores.append(dice)
            validation_accuracies.append(accuracy)

            if dice > max_dice:
                max_dice = dice
                best_iter = t

                torch.save(model.state_dict(), save_name)
            model.train()

    end_time = time.time()
    training_time = end_time - start_time
    training_losses_str = str(training_losses)
    validation_losses_str = str(validation_losses)
    validation_dice_scores_str = str(validation_dice_scores)
    validation_accuracies_str = str(validation_accuracies)

    plot_dir = f"results/{algorithm}/{model_name}"
    os.makedirs(plot_dir, exist_ok=True)

    # Define the metrics to plot
    metrics = {
        "Loss Curve":       (training_losses,      'tab:purple',  'Training Loss'),
        "Validation Curve": (validation_losses,    'tab:pink',   'Validation Loss'),
        "Epoch Time Curve": (iteration_train_times,'tab:cyan','Epoch Time (s)'),
    }

    for title, (data, color, ylabel) in metrics.items():
        plt.figure(figsize=(7, 4))
        epochs = list(range(1, len(data) + 1))
        plt.plot(
            epochs, data,
            marker='D',
            linestyle='-.',
            color=color
        )
        plt.title(f"{algorithm} – {title}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle=':', alpha=0.6)
        filename = f"{algorithm.lower().replace(' ', '_')}_{title.lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename), dpi=150)
        plt.close()

    save_results_to_csv(
        file_name=file_name,
        batch_size=batch_size,
        epochs=iterations,
        learning_rate=learning_rate,
        noise_multiplier=None,
        max_per_sample_grad_norm=None,
        clipping_mode=None,
        algorithm=algorithm,
        overall_privacy_spent=None,
        dataset=dataset,
        model_name=model_name,
        device=device,
        iteration_train_times=iteration_train_times,
        training_losses=training_losses,
        validation_losses=validation_losses,
        validation_dice_scores=validation_dice_scores,
        validation_accuracies=validation_accuracies,
        total_training_time=training_time,
    )
    print("Best iteration: ", best_iter, "Best val dice: ", max_dice)
    return model


if __name__ == "__main__":
    args = argument_parser().parse_args()
    print(args)
    set_seed(args.seed)

    train(args)
