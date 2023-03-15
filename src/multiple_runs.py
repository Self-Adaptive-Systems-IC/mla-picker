#!/usr/bin/env python
from main import main
import time

def multiple_run():
    files = ['src/data/gender.pkl', 'src/data/outlier.pkl', 'src/data/titanic.pkl']
    for e in files:
        start = time.time()
        main(e)
        end = time.time()
        print(f"Elapsed {end-start}s")

 
if __name__ == "__main__":
    multiple_run()
