import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("module_dir", type=str, help="python module directory")
parser.add_argument("module_name", type=str, help="module name")

args = parser.parse_args()


if __name__ == "__main__":
    module_path = os.path.join(args.module_dir, args.module_name)

    if not os.path.exists(module_path):
        os.makedirs(module_path)

    module_init_filename = os.path.join(module_path, "__init__.py")

    content = f"import {args.module_name}._{args.module_name}_internal"
    write_file = True
    if os.path.exists(module_init_filename):
        with open(module_init_filename, "r") as f:
            if f.read() == content:
                write_file = False

    if write_file:
        with open(module_init_filename, "w") as f:
            f.write(content)
