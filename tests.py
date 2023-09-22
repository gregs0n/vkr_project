from datetime import datetime
from os import mkdir
from multiprocessing import Process

from scheme import BalanceScheme
from enviroment import *
from boundary import getBoundary
from custom_numerics.wraps import timer

@timer
def runtest(test: Test, folder=""):
    print(f"{datetime.now().strftime('%H:%M:%S')} - |[ {test.test_no:03} ]| started {getTestName(test)}")
    u0 = None
    eps = 1e-3
    log = 1
    if 1 and test.cells[0] > 1:
        if (test.cell_size <= 17):
            acells = [test.cells[0]//2, test.cells[1]//2]
        else: acells = test.cells
        aCellSize = 9

        assumpTest = Test(
            0, test.bnd, test.material, cells=acells, cell_size=aCellSize
        )
        params = [
            *getBoundary(assumpTest, f_off=0, g_off=0),
            assumpTest.material,
            assumpTest.limits,
            assumpTest.cells,
            aCellSize,
            getTestName(test)+"_Assump" ,
            folder,
            log
        ]
        assump = BalanceScheme(*params)
        u0 = assump.Compute(eps)

    params = [
        *getBoundary(test, f_off=0, g_off=0),
        test.material,
        test.limits,
        test.cells,
        test.cell_size,
        getTestName(test),
        folder,
        log
    ]

    s = BalanceScheme(*params)
    s.Compute(eps, u0)
    s.save()
    print(f"{datetime.now().strftime('%H:%M:%S')} - |[ {test.test_no:03} ]|    over {s.test_name}")

cells = [16, 16]
cell_size = 17


def SingleTest():
    folder = InitFolder("SingleTest")
    material = materials[1]._replace(thermal_cond=25.0)
    test = Test(1, -4, material, cells, cell_size)

    runtest(test, folder)


def TestMaterials():
    folder = InitFolder("TestMaterials")
    test = Test(0, -2, materials[0], cells, cell_size)
    procs = []
    for i in range(6):
        test = Test(14+i, -2, materials[i], cells, cell_size)
        runtest(test, folder); continue
        proc = Process(target=runtest, args=(test, folder))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    
def TestSquares():
    folder = InitFolder("TestSquares")
    material = materials[1]._replace(thermal_cond=10.0)
    test = Test(0, -2, material, cells, cell_size)
    procs = []
    _cells = [1, 2, 4, 8, 16]
    step = 256
    for i in range(len(_cells)):
        test = Test(32+i, -1, material, [_cells[i], _cells[i]], (step//_cells[i])+1)
        runtest(test, folder); continue
        proc = Process(target=runtest, args=(test, folder))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()


TCC = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 25, 50, 100, 200, 400]


def TestThermalCond():
    folder = InitFolder("TestThermalCond")
    material = materials[1]._replace(thermal_cond=TCC[0])
    test = Test(10, -1, material, cells, cell_size)
    procs = []
    for i in range(len(TCC)-1):
        material = materials[1]._replace(thermal_cond=TCC[i])
        test = Test(2+i, -1, material, cells, cell_size)
        runtest(test, folder); continue
        proc = Process(target=runtest, args=(test, folder))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()


def TestMyFunctions():
    folder = InitFolder("TestMyFunctions")
    test = Test(40, 0, materials[1], cells, cell_size)
    procs = []
    for i in range(12):
        test = Test(20 + i, i - 6, materials[1], cells, cell_size)
        runtest(test, folder); continue
        proc = Process(target=runtest, args=(test, folder))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()


def InitFolder(folder_name: str):
    now = datetime.now()
    time_string = str(datetime.date(now)) + now.strftime("_%H-%M-%S")
    folder = f"{time_string}_{folder_name}_[{cells[0]}_{cells[1]}]_{cell_size:02d}"
    mkdir(folder)
    mkdir(folder + "/CG")
    mkdir(folder + "/logs")
    mkdir(folder + "/bin")
    return folder
