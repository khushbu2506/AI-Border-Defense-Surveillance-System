import os

label_dir = "dataset/train/labels"

for file in os.listdir(label_dir):

    path = os.path.join(label_dir, file)

    if not file.endswith(".txt"):
        continue

    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        cls = int(parts[0])

        if cls <= 2:
            new_lines.append(line)

    if len(new_lines) == 0:
        os.remove(path)
    else:
        with open(path, "w") as f:
            f.writelines(new_lines)

print("Labels cleaned successfully")