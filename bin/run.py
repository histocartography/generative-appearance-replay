import argparse
import json
from pathlib import Path

import torch

from source.dataset.dataset import MnMsDataset, ODSDataset, MSProstateDataset
from source.trainers.garda import GarDATrainer
from source.config.config import Config
from source.utils.misc import seed_everything


def main(args, task, data_path, device):
    """Run training or testing.

    Args:
        args: train/test configuration
        task: segmentation task (ods/cs/ps)
        data_path: base path to datasets
        device: gpu or cpu device
    """
    seed_everything(seed=args.seed, device=device)
    test_dataset_prev = None
    test_dataset_src = None

    # optic disc segmentation (ods)
    if task == "ods":
        curr_csv_p = f"{data_path}/{args.curr_dataset}/{args.curr_dataset.lower()}_all_slides.csv"
        curr_split_p = f"{data_path}/{args.curr_dataset}/splits/split_{args.curr_split}.csv"
        curr_img_p = f"{data_path}/{args.curr_dataset}/preprocessed/images"
        curr_anno_p = f"{data_path}/{args.curr_dataset}/preprocessed/masks"
        curr_mask_p = f"{data_path}/{args.curr_dataset}/preprocessed/background_masks"

        gan_dataset = ODSDataset(
            csv_path=curr_csv_p,
            mode="train",
            split_path=curr_split_p,
            train_gan=True,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
            mask_path=curr_mask_p,
            augmentation=args.augmentation,
        )
        train_dataset = ODSDataset(
            csv_path=curr_csv_p,
            mode="train",
            split_path=curr_split_p,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
            mask_path=curr_mask_p,
            augmentation=args.augmentation,
        )
        val_dataset = ODSDataset(
            csv_path=curr_csv_p,
            mode="val",
            split_path=curr_split_p,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
            mask_path=curr_mask_p,
        )
        test_dataset = ODSDataset(
            csv_path=curr_csv_p,
            mode="test",
            split_path=curr_split_p,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
            mask_path=curr_mask_p,
            dataset=args.curr_dataset,
            split=args.curr_split,
        )
        if args.prev_dataset is not None:
            test_dataset_prev = ODSDataset(
                csv_path=f"{data_path}/{args.prev_dataset}/{args.prev_dataset.lower()}_all_slides.csv",
                mode="test",
                split_path=f"{data_path}/{args.prev_dataset}/splits/split_{args.prev_split}.csv",
                image_path=f"{data_path}/{args.prev_dataset}/preprocessed/images",
                anno_path=f"{data_path}/{args.prev_dataset}/preprocessed/masks",
                mask_path=f"{data_path}/{args.prev_dataset}/preprocess/background_masks",
                dataset=args.prev_dataset,
                split=args.prev_split,
            )
        if args.src_dataset is not None:
            test_dataset_src = ODSDataset(
                csv_path=f"{data_path}/{args.src_dataset}/{args.src_dataset.lower()}_all_slides.csv",
                mode="test",
                split_path=f"{data_path}/{args.src_dataset}/splits/split_{args.src_split}.csv",
                image_path=f"{data_path}/{args.src_dataset}/preprocessed/images",
                anno_path=f"{data_path}/{args.src_dataset}/preprocessed/masks",
                mask_path=f"{data_path}/{args.src_dataset}/preprocess/background_masks",
                dataset=args.src_dataset,
                split=args.src_split,
            )
    # cardiac segmentation (cs)
    elif task == "cs":
        curr_csv_p = f"{data_path}/preprocessed/labeled_{args.curr_dataset}_-1/all_slides.csv"
        curr_split_p = f"{data_path}/preprocessed/labeled_{args.curr_dataset}_-1/split.csv"
        curr_img_p = f"{data_path}/preprocessed/labeled_{args.curr_dataset}_-1/images"
        curr_anno_p = f"{data_path}/preprocessed/labeled_{args.curr_dataset}_-1/masks"

        gan_dataset = MnMsDataset(
            csv_path=curr_csv_p,
            mode="train",
            split_path=curr_split_p,
            train_gan=True,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
            augmentation=args.augmentation,
        )
        train_dataset = MnMsDataset(
            csv_path=curr_csv_p,
            mode="train",
            split_path=curr_split_p,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
            augmentation=args.augmentation,
        )
        val_dataset = MnMsDataset(
            csv_path=curr_csv_p,
            mode="val",
            split_path=curr_split_p,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
        )
        test_dataset = MnMsDataset(
            csv_path=curr_csv_p,
            mode="test",
            split_path=curr_split_p,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
            dataset=args.curr_dataset,
            split=args.curr_split,
            info_path="{data_path}/partition/mnms_dataset_info.csv",
        )
        if args.prev_dataset is not None:
            test_dataset_prev = MnMsDataset(
                csv_path=f"{data_path}/preprocessed/labeled_{args.prev_dataset}_-1/all_slides.csv",
                mode="test",
                split_path=f"{data_path}/preprocessed/labeled_{args.prev_dataset}_-1/split.csv",
                image_path=f"{data_path}/preprocessed/labeled_{args.prev_dataset}_-1/images",
                anno_path=f"{data_path}/preprocessed/labeled_{args.prev_dataset}_-1/masks",
                dataset=args.prev_dataset,
                split=args.prev_split,
                info_path="{data_path}/partition/mnms_dataset_info.csv",
            )
        if args.src_dataset is not None:
            test_dataset_src = MnMsDataset(
                csv_path=f"{data_path}/preprocessed/labeled_{args.src_dataset}_-1/all_slides.csv",
                mode="test",
                split_path=f"{data_path}/preprocessed/labeled_{args.src_dataset}_-1/split.csv",
                image_path=f"{data_path}/preprocessed/labeled_{args.src_dataset}_-1/images",
                anno_path=f"{data_path}/preprocessed/labeled_{args.src_dataset}_-1/masks",
                dataset=args.src_dataset,
                split=args.src_split,
                info_path="{data_path}/partition/mnms_dataset_info.csv",
            )
    # prostate segmentation (ps)
    elif task == "ps":
        curr_csv_p = f"{data_path}/preprocessed/{args.curr_dataset}/all_slides.csv"
        curr_split_p = f"{data_path}/preprocessed/{args.curr_dataset}/split.csv"
        curr_img_p = f"{data_path}/preprocessed/{args.curr_dataset}/images"
        curr_anno_p = f"{data_path}/preprocessed/{args.curr_dataset}/masks"

        gan_dataset = MSProstateDataset(
            csv_path=curr_csv_p,
            mode="train",
            split_path=curr_split_p,
            train_gan=True,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
            augmentation=args.augmentation,
        )
        train_dataset = MSProstateDataset(
            csv_path=curr_csv_p,
            mode="train",
            split_path=curr_split_p,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
            augmentation=args.augmentation,
        )
        val_dataset = MSProstateDataset(
            csv_path=curr_csv_p,
            mode="val",
            split_path=curr_split_p,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
        )
        test_dataset = MSProstateDataset(
            csv_path=curr_csv_p,
            mode="test",
            split_path=curr_split_p,
            image_path=curr_img_p,
            anno_path=curr_anno_p,
            dataset=args.curr_dataset,
            split=args.curr_split,
        )

        if args.prev_dataset is not None:
            test_dataset_prev = []
            for prev_ds in args.prev_dataset:
                test_dataset_prev.append(
                    MSProstateDataset(
                        csv_path=f"{data_path}/preprocessed/{prev_ds}/all_slides.csv",
                        mode="test",
                        split_path=f"{data_path}/preprocessed/{prev_ds}/split.csv",
                        image_path=f"{data_path}/preprocessed/{prev_ds}/images",
                        anno_path=f"{data_path}/preprocessed/{prev_ds}/masks",
                        dataset=prev_ds,
                        split=args.prev_split,
                    )
                )

        if args.src_dataset is not None:
            test_dataset_src = MSProstateDataset(
                csv_path=f"{data_path}/preprocessed/{args.src_dataset}/all_slides.csv",
                mode="test",
                split_path=f"{data_path}/preprocessed/{args.src_dataset}/split.csv",
                image_path=f"{data_path}/preprocessed/{args.src_dataset}/images",
                anno_path=f"{data_path}/preprocessed/{args.src_dataset}/masks",
                dataset=args.src_dataset,
                split=args.src_split,
            )
    else:
        raise ValueError(f"Task '{task}' is not implemented.")

    # initialize trainer
    datasets = (
        gan_dataset,
        train_dataset,
        val_dataset,
        test_dataset,
        test_dataset_prev,
        test_dataset_src,
    )
    trainer = GarDATrainer(args=args, task=task, datasets=datasets)

    # train or test
    assert args.task in ["train_seg", "train_gen", "test_seg"], "Task not supported"
    if args.task == "train_seg":
        trainer.train_seg()
    elif args.task == "train_gen":
        trainer.train_gen()
    elif args.task == "test_seg":
        trainer.test_seg()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurations for UDA Segmentation")
    parser.add_argument("--config_path", type=str, help="path to configuration file")
    parser.add_argument("--data_path", type=str, help="base path to all datasets")
    parser.add_argument(
        "--task",
        type=str,
        choices=["ods", "cs", "ps"],
        help="segmentation task (ods/cs/ps)",
    )
    args = parser.parse_args()

    # load config file
    with open(args.config_path, "r") as ifile:
        config = Config(json.load(ifile))

    # create output folder if it doesn't exist yet
    Path(config.save_path).mkdir(parents=True, exist_ok=True)

    # set device to cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run training or testing
    main(config, task=args.task, data_path=args.data_path, device=device)
    print("Done!")
