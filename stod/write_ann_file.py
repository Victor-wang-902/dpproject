from pathlib import Path

if __name__ == "__main__":
    Path("temp_Ann").mkdir(parents=True, exist_ok=True)
    with open("temp_Ann/train.txt", "w") as f:
        for i in range(1,30001):
            f.write(str(i) + "\n")
    with open("temp_Ann/val.txt", "w") as f:
        for i in range(30001, 50001):
            f.write(str(i) + "\n")