#!/usr/bin/env python
from main import main

def multiple_run():
    files = ['src/data/gender.pkl', 'src/data/outlier.pkl', 'src/data/titanic.pkl']
    for e in files:
        main(e)

 
if __name__ == "__main__":
    multiple_run()
