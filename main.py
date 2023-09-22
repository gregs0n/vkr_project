from tests import *
from multiprocessing import Process
import os
import pickle
from custom_numerics.draw import draw1D, drawHeatmap


def main():
    tests = [SingleTest, TestThermalCond, TestMaterials, TestMyFunctions, TestSquares]
    procs = []
    for test in tests[:1]:
        test(); continue
        proc = Process(target=test)
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()


def DrawTests(interactive: bool):
    dirs = list(filter(lambda dir: dir.startswith('2023-06-14'), os.listdir()))
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
    dirs = list(filter(lambda dir: dir.endswith('.bin'), os.listdir(selected_dir+"/bin")))
    for (i, elem) in enumerate(dirs):
        print(f"\t{i:02}. " + elem)
    test_no = input("Select a test_no:" )
    while not test_no.isnumeric() and int(test_no) > len(dirs):
        test_no = int(input("Select a test_no:" ))
    test_no = int(test_no)
    print(f"Selected test: {test_no}. '{dirs[test_no]}'")
    file = open(f"{selected_dir}/bin/{dirs[test_no]}", "rb")
    scheme = pickle.load(file)
    file.close()
    if (interactive):
        return scheme
    show_plot = 0
    for i in range(3):
        scheme.show_res(code=i, show_plot=show_plot)
    scheme.trace_newt_err(show_plot)
    scheme.trace_cg_err(show_plot)

def DrawAll():
    dirs = list(filter(lambda dir: dir.startswith('2023-06-14'), os.listdir()))
    for (i, _dir) in enumerate(dirs):
        print(f"\n[{i:02}] FOLDER::{_dir}")
        bins = list(filter(lambda dir: dir.endswith('.bin'), os.listdir(_dir + "/bin")))
        for (j, bin) in enumerate(bins):
            print(f"\t[{i:02}|{j:02}] FILE::{bin}")
            file = open(f"{_dir}/bin/{bin}", "rb")
            scheme = pickle.load(file)
            file.close()
            show_plot = 0
            scheme.show_res(code=0, show_plot=show_plot)
            if (j == 0):
                drawHeatmap(scheme.F[0, 0], [0, 1], "F(x, y)", show_plot=0)
            scheme.trace_newt_err(show_plot)
            scheme.trace_cg_err(show_plot)

if __name__ == "__main__":
    #main()
    TestSquares()
    #DrawTests(0)
    #DrawAll()
