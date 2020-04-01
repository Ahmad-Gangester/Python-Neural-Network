from Classes.Trainer import Trainer
import os
from pathlib import Path

def main():
    cur_path=Path(__file__).parent.absolute()
    print(str(cur_path))
    Ai=Trainer(str(cur_path) + "\\Data")
    Ai.Train()

if(__name__ == "__main__"):
    main()
