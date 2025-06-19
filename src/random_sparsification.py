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
from opacus import PrivacyEngine
from utils import per_class_dice

from data import get_data


def argument_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument(
        "--num_iterations", default=5, type=int, help="Number of Epochs"
    )
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--n_classes", default=9, type=int)
    parser.add_argument("--ffc_lambda", default=0, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--batch_partitions",
        default=2,
        help="Partition data of a batch to reduce memory footprint.",
        type=int,
    )


    parser.add_argument(
        "--noise_multiplier",
        default=0.5,
        type=float,
        help="Level of independent Gaussian noise into the gradient",
    )
    parser.add_argument(
        "--target_delta", default=1e-5, type=float, help="Target privacy budget δ"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=[1.0] * 122,  # 1.0 for flat clipping mode and [1.0] * 122 for NestedUNet
        type=float,
        help="Per-sample gradient clipping threshold",
    )
    parser.add_argument(
        "--clipping_mode",
        default="per_layer",
        choices=["flat", "per_layer", "adaptive"],
        help="Gradient clipping mode",
    )

    parser.add_argument(
        "--final_rate",
        default=0.9,
        type=float,
        help="Sparsification rate at the end of gradual cooling.",
    )
    parser.add_argument(
        "--refresh",
        default=2,
        type=int,
        help="Randomization times of sparsification mask per epoch.",
    )


    parser.add_argument("--dataset", default="Pneumothorax", choices=["Pneumothorax"])
    parser.add_argument("--image_size", default="224", type=int)
    parser.add_argument(
        "--image_dir",
        default="data",
        choices=["data"],
    )
    parser.add_argument("--model_name", default="NestedUNet", choices=["unet", "NestedUNet"])


    parser.add_argument("--g_ratio", default=0.5, type=float)


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


def calculate_exit_rate(
    current_iteration, total_iterations, refresh_rate, final_rate, train_loader_length
):
    if total_iterations <= 1:
        return 0
    return np.clip(
        final_rate
        * (
            current_iteration * refresh_rate
            + total_iterations // (train_loader_length // refresh_rate)
        )
        / (refresh_rate * total_iterations - 1),
        0,
        final_rate,
    )


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
    im_save=False,
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


        correct_pixels += (idx == label.squeeze(1)).sum().item()
        total_pixels += torch.numel(label.squeeze(1))

        if im_save:

            name = f"{algorithm}-{counter}"
            fig, ax = plt.subplots(1, 2)
            fig.suptitle(name, fontsize=10)

            ax[0].imshow(label.cpu().squeeze().numpy(), cmap="gray")
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
    return dice, loss, accuracy


def train(args):

    batch_size = args.batch_size
    iterations = args.num_iterations
    learning_rate = args.learning_rate
    n_classes = args.n_classes


    noise_multiplier = args.noise_multiplier
    target_delta = args.target_delta
    max_grad_norm = args.max_grad_norm
    clipping_mode = args.clipping_mode
    algorithm = "Opacus-RS"


    final_rate = args.final_rate
    refresh = args.refresh


    dataset = args.dataset
    img_size = args.image_size
    data_path = args.image_dir
    model_name = args.model_name


    ratio = args.g_ratio


    device = args.device

    training_losses = []
    validation_losses = []
    validation_dice_scores = []
    validation_accuracies = []
    criterion_seg = CombinedLoss()
    criterion_ffc = FocalFrequencyLoss()
    save_name = f"results/{algorithm}/{model_name}_{noise_multiplier}.pt"
    file_name = f"results/{algorithm}/{model_name}.csv"
    location = f"{noise_multiplier}"

    max_dice = 0

    best_iter = 0

    model = get_model(model_name, ratio=ratio, num_classes=n_classes).to(device)
    model.train()
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = (
        get_data(data_path, img_size, batch_size)
    )
    optimizer = torch.optim.SGD(
        list(model.parameters()), lr=learning_rate, weight_decay=args.weight_decay
    )

    privacy_engine = PrivacyEngine()
    model, optimizer, data_path = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        clipping=clipping_mode,
    )

    num_params = sum(p.numel() for p in model.parameters())
    start_time = time.time()
    iteration_train_times = []
    overall_privacy_spent = []
    train_epoch_losses = []
    mini_batch = 0
    for t in range(iterations):
        start_epoch_time = time.time()

        total_loss = 0
        total_samples = 0


        gradient = torch.zeros(size=[num_params]).to(device)

        for img, label in tqdm.tqdm(train_loader):
            current_iteration = t * len(train_loader) + mini_batch
            refresh_interval = len(train_loader) // args.refresh


            if current_iteration % refresh_interval == 0:
                rate = calculate_exit_rate(
                    t, iterations, args.refresh, args.final_rate, len(train_loader)
                )
            else:
                if "rate" not in locals():
                    rate = 0

            mask = torch.randperm(num_params, device=device, dtype=torch.long)[
                : int(rate * num_params)
            ]

            img = img.to(device)
            label = label.to(device)
            label_oh = torch.nn.functional.one_hot(
                label, num_classes=n_classes
            ).squeeze()


            optimizer.zero_grad()


            pred = model(img)
            max_val, idx = torch.max(pred, 1)
            pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
            pred_oh = pred_oh.permute(0, 3, 1, 2)
            label_oh = label_oh.permute(0, 3, 1, 2)
            loss = criterion_seg(
                pred, label.squeeze(1), device=device
            ) + args.ffc_lambda * criterion_ffc(pred_oh, label_oh)

            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            batch_grad = torch.zeros(
                sum(p.numel() for p in model.parameters()), device=device
            )

            with torch.no_grad():
                offset = 0
                for grad in grads:
                    if grad is not None:
                        numel = grad.numel()
                        batch_grad[offset : offset + numel] = grad.reshape(-1)
                        offset += numel


            batch_grad[mask] = 0
            norm = torch.norm(batch_grad, dim=0)
            max_grad_norm_value = (
                max_grad_norm[0] if isinstance(max_grad_norm, list) else max_grad_norm
            )
            scale = torch.clamp(max_grad_norm_value / norm, max=1)
            gradient += (batch_grad * scale.view(-1, 1)).sum(dim=0)


            mini_batch += 1
            if mini_batch == args.batch_partitions:
                gradient = gradient / args.batch_size
                mini_batch = 0


                noise = torch.normal(
                    0,
                    noise_multiplier * max_grad_norm_value / args.batch_size,
                    size=gradient.shape,
                ).to(device)
                noise[mask] = 0
                gradient += noise


                offset = 0
                for p in model.parameters():
                    if p.grad is not None:
                        shape = p.grad.shape
                        numel = p.grad.numel()
                        p.grad.data = gradient[offset : offset + numel].view(shape)
                        offset += numel

                optimizer.step()
                for p in model.parameters():
                    p.grad = None
                gradient = torch.zeros(size=[num_params]).to(device)

            total_loss = loss.item() + img.size(0)
            total_samples += img.size(0)

        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time
        iteration_train_times.append(epoch_time)
        average_loss = total_loss / total_samples
        training_losses.append(average_loss)
        train_epoch_losses.append(loss.item())


        privacy_spent = privacy_engine.get_epsilon(delta=target_delta)
        overall_privacy_spent.append(privacy_spent)

        if t % 10 == 0 or t > 4:
            dice, validation_loss, accuracy = eval(
                val_loader,
                criterion_seg,
                model,
                dice_s=True,
                n_classes=n_classes,
                dataset=dataset,
                algorithm=algorithm,
                location=f"{model_name}/{location}",
                im_save=True,
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

    plot_dir = f"results/{algorithm}/{model_name}/{location}"
    os.makedirs(plot_dir, exist_ok=True)


    metrics = {
        "RS-DP Curve":        (overall_privacy_spent,  'tab:purple', 'Privacy ε'),
        "Batch Loss Trend":   (training_losses,        'tab:orange', 'Avg Loss'),
        "Epoch Loss Curve":   (train_epoch_losses,     'tab:cyan',   'Epoch Loss'),
    }

    for title, (data, color, ylabel) in metrics.items():
        plt.figure(figsize=(7, 4))
        epochs = list(range(1, len(data) + 1))
        plt.plot(epochs, data,
                 marker='s',
                 linestyle='--',
                 color=color)
        plt.title(f"{algorithm} | {title}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        fn = f"{algorithm.lower().replace(' ', '_')}_{title.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(plot_dir, fn), dpi=150)
        plt.close()

    save_results_to_csv(

        file_name=file_name,

        batch_size=batch_size,
        epochs=iterations,
        learning_rate=learning_rate,

        noise_multiplier=noise_multiplier,

        max_grad_norm=max_grad_norm,
        clipping_mode=clipping_mode,
        algorithm=algorithm,
        overall_privacy_spent=overall_privacy_spent,

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
    return model


if __name__ == "__main__":
    args = argument_parser().parse_args()
    set_seed(args.seed)

    train(args)
