from scheme import BalanceScheme
from enviroment import *
from datetime import datetime
from boundary import getBoundary
from os import mkdir
from multiprocessing import Process

def runtest(test: Test, folder=''):
    print('-'*98)
    u0 = None
    aruntime = 0
    eps = 1e-3
    if 1:
        aCells = 5
        aCellSize = 6
        assumpTest = Test(0, test.bnd, test.material,
                          cells=[aCells, aCells], cell_size=aCellSize)
        params = [*getBoundary(assumpTest, f_off=0, g_off=0),
                  assumpTest.material,
                  assumpTest.limits,
                  assumpTest.cells, aCellSize,
                  getTestName(assumpTest)]
        assump = BalanceScheme(*params)
        print("Assumption for: " + getTestName(test))
        u0, aruntime = assump.Compute(eps)
        #assump.show_res(code=0, show_plot=1)
    
    runtime = 0
    params = [*getBoundary(test, f_off=0, g_off=0),
              test.material,
              test.limits, test.cells, test.cell_size,
              getTestName(test), folder]
    s = BalanceScheme(*params)
    #s.show_res(0, show_plot=1)
    #s.show_res(2, show_plot=1)
    print(s.test_name)
    _, runtime = s.Compute(eps, u0)
    s.trace_newt_err(show_plot=0)
    s.trace_cg_err(show_plot=0)
    s.show_res(show_plot=0)
    return runtime, aruntime

cells = [5, 5]
cell_size = 51

def SingleTest():
    folder = InitFolder("SingleTest")
    t1, t2 = 0, 0
    filename = 'logs/SingleTest_log.txt'
    material = materials[1]._replace(thermal_cond=TCC[-1])
    test = Test(666, -1, material, cells, cell_size)

    LogTest(filename, 0, test)
    t1, t2 = runtest(test, folder)
    #proc = Process(target=runtest, args=(test, folder))
    LogTest(filename, 1, t1, t2)
    LogTest(filename, 2)

def TestMaterials():
    folder = InitFolder("TestMaterials")
    t1, t2 = 0, 0
    filename = 'logs/TestMaterials_log.txt'
    test = Test(0, -1, materials[0], cells, cell_size)
    LogTest(filename, 0, test)
    procs = []
    for i in range(6):
        test = Test(i, -1, materials[i], cells, cell_size)
        proc = Process(target=runtest, args=(test, folder))
        procs.append(proc)
        proc.start()
        LogTest(filename, 1, t1, t2)
    for proc in procs: proc.join()
    LogTest(filename, 2)

TCC = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 25, 50, 100, 250, 500]
def TestThermalCond():
    folder = InitFolder("TestThermalCond")
    t1, t2 = 0, 0
    filename = 'logs/TestThermalCond_log.txt'
    material = materials[1]._replace(thermal_cond=TCC[0])
    test = Test(10, -1, material, cells, cell_size)
    LogTest(filename, 0, test)
    procs = []
    for i in range(3, len(TCC), 2):
        material = materials[1]._replace(thermal_cond=TCC[i])
        test = Test(10+i, -1, material, cells, cell_size)
        proc = Process(target=runtest, args=(test, folder))
        procs.append(proc)
        proc.start()
        LogTest(filename, 1, t1, t2)
    for proc in procs: proc.join()
    LogTest(filename, 2)

def TestMyFunctions():
    folder = InitFolder("TestMyFunctions")
    t1, t2 = 0, 0
    filename = 'logs/TestMyFunctions_log.txt'
    test = Test(40, 0, materials[1], cells, cell_size)
    LogTest(filename, 0, test)
    procs = []
    for i in range(10):
        test = Test(40+i, i-4, materials[1], cells, cell_size)
        proc = Process(target=runtest, args=(test, folder))
        procs.append(proc)
        proc.start()
        LogTest(filename, 1, t1, t2)
    for proc in procs: proc.join()
    LogTest(filename, 2)

def LogTest(fname: str, cmd: int, *args):
    file = open(fname, 'a')
    now = datetime.now()
    time_string = str(datetime.date(now)) + now.strftime(' %H:%M:%S')
    if cmd == 0:
        print(f"\nTest '{getTestName(args[0])}'", file=file)
        print(f"Started - {time_string}", file=file)
    elif cmd == 1:
        print(f"{args[0]:.6f}, {args[1]:.6f}", file=file)
    elif cmd == 2:
        print(f"Test's over - {time_string}", file=file)
    file.close()

def InitFolder(folder_name: str):
    now = datetime.now()
    time_string = str(datetime.date(now)) + now.strftime('_%H-%M-%S')
    folder = f"{time_string}_{folder_name}"
    mkdir(folder)
    mkdir(folder+"/CG")
    return folder