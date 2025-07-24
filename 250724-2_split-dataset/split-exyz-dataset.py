"""
python split_xyz.py --input dataset.xyz --test_ratio 0.1 --train_output train.xyz --test_output test.xyz
"""

import argparse
import random
from ase.io import read, write
import math

# ----------- 参数解析 -----------
parser = argparse.ArgumentParser(
    description="Split extended XYZ file into training and test sets.\n\n"
                "Example usage:\n"
                "  python split_xyz.py --input dataset.xyz --test_ratio 0.1 --train_output train.xyz --test_output test.xyz\n",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("--input", type=str, default="dataset.xyz", help="Path to input extended XYZ file")
parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of configurations to use for test set")
parser.add_argument("--train_output", type=str, default="train.xyz", help="Output path for training set")
parser.add_argument("--test_output", type=str, default="test.xyz", help="Output path for test set")
args = parser.parse_args()

# ----------- 读取结构 -----------
all_structures = read(args.input, index=":")
total = len(all_structures)

# ----------- 抽样并划分 -----------
test_size = math.ceil(total * args.test_ratio)
all_indices = list(range(total))
random.shuffle(all_indices)
test_indices = set(all_indices[:test_size])

train_structures = [atoms for i, atoms in enumerate(all_structures) if i not in test_indices]
test_structures  = [atoms for i, atoms in enumerate(all_structures) if i in test_indices]

# ----------- 写入文件 -----------
write(args.train_output, train_structures, format="extxyz")
write(args.test_output, test_structures, format="extxyz")

print(f"Total structures: {total}")
print(f"Train set: {len(train_structures)} structures written to {args.train_output}")
print(f"Test set: {len(test_structures)} structures written to {args.test_output}")
