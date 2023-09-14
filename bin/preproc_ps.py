import argparse
import numpy as np

from source.dataset.preprocessing import MSProstatePreprocessor


def main(in_path, out_path):
    np.random.seed(24)
    processor = MSProstatePreprocessor(base_in=in_path, base_out=out_path)
    processor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration to preprocess prostate segmentation data.")
    parser.add_argument("--in_path", type=str, help="base input path to (raw) data")
    parser.add_argument("--out_path", type=str, help="base output path for preprocessed data")
    args = parser.parse_args()

    main(in_path=args.in_path, out_path=args.out_path)
    print("Done!")
