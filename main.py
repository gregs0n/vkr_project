from tests import *
from multiprocessing import Process
import os
import pickle


def main():
    tests = [SingleTest, TestThermalCond, TestMaterials, TestMyFunctions]
    procs = []
    for test in tests:
        proc = Process(target=test)
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def DrawTests():
    dirs = list(filter(lambda dir: dir.startswith('2023'), os.listdir()))
    print('\n'.join(dirs))
    selected_dir = input("Select test's dir: ")
    while selected_dir not in dirs and not selected_dir.isnumeric():
        if selected_dir.isnumeric() and int(selected_dir) < len(dirs):
            selected_dir = dirs[int(selected_dir)]
            break
        print("dir does not exists")
        selected_dir = input("Select test's dir: ")
    if selected_dir.isnumeric():
        selected_dir = dirs[int(selected_dir)]
    print(f"Selected dir: {selected_dir}")
    dirs = list(filter(lambda dir: dir.endswith('.bin'), os.listdir(selected_dir)))
    for (i, elem) in enumerate(dirs):
        if elem.endswith(".bin"):
            print(f"\t{i:02}. " + elem)
    test_no = input("Select a test_no:" )
    while not test_no.isnumeric() and int(test_no) > len(dirs):
        test_no = int(input("Select a test_no:" ))
    test_no = int(test_no)
    print(f"Selected test: {test_no}. '{dirs[test_no]}'")
    file = open(f"{selected_dir}/{dirs[test_no]}", "rb")
    scheme = pickle.load(file)
    show_plot = 0
    for i in range(3):
        scheme.show_res(code=i, show_plot=show_plot)
    scheme.trace_newt_err(show_plot)
    scheme.trace_cg_err(show_plot)

if __name__ == "__main__":
    main()
    #DrawTests()
