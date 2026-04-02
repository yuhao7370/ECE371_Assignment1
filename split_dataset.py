from pathlib import Path
import random
import shutil


source_dir = Path("data/UCMerced_LandUse/Images")
output_dir = Path("data/uc_merced_dataset")
train_ratio = 0.8
seed = 2026


def main() -> None:
    random.seed(seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    (output_dir / "train").mkdir(parents=True)
    (output_dir / "val").mkdir(parents=True)

    class_names = []
    train_lines = []
    val_lines = []
    train_total = 0
    val_total = 0

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        class_names.append(class_name)

        images = [path for path in class_dir.iterdir() if path.is_file()]
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        train_class_dir = output_dir / "train" / class_name
        val_class_dir = output_dir / "val" / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in train_images:
            shutil.copy2(image_path, train_class_dir / image_path.name)
            train_lines.append(f"train/{class_name}/{image_path.name}")

        for image_path in val_images:
            shutil.copy2(image_path, val_class_dir / image_path.name)
            val_lines.append(f"val/{class_name}/{image_path.name}")

        train_total += len(train_images)
        val_total += len(val_images)

        # print(f"{class_name}: train={len(train_images)}, val={len(val_images)}")

    (output_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (output_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (output_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print()
    print(f"classes: {len(class_names)}")
    print(f"train images: {train_total}")
    print(f"val images: {val_total}")
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
