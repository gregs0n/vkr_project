from tests import *
from multiprocessing import Process


def main():
    tests = [TestThermalCond, TestMyFunctions, TestMaterials, SingleTest]
    tests[0]()
    # procs = []
    # for test in tests:
    #    proc = Process(target=test)
    #    procs.append(proc)
    #    proc.start()
    #
    # for proc in procs:
    #    proc.join()


if __name__ == "__main__":
    main()
