from datetime import datetime
from os import mkdir
from multiprocessing import Process

from scheme import BalanceScheme
from enviroment import *
from boundary import getBoundary


def runtest(test: Test, folder=""):
    u0 = None
    eps = 1e-3
    log = 1
    if 1:
        aCells = 8
        aCellSize = 9
        assumpTest = Test(
            0, test.bnd, test.material, cells=[aCells, aCells], cell_size=aCellSize
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
    print(f"{datetime.now().strftime('%H:%M:%S')} - |[ {test.test_no:03} ]| started")
    s.Compute(eps, u0)
    print(f"{datetime.now().strftime('%H:%M:%S')} - |[ {test.test_no:03} ]| over")
    s.save()

cells = [8, 8]
cell_size = 65


def SingleTest():
    folder = InitFolder("SingleTest")
    material = materials[1]._replace(thermal_cond=TCC[4])
    test = Test(1, -6, material, cells, cell_size)

    runtest(test, folder)


def TestMaterials():
    folder = InitFolder("TestMaterials")
    test = Test(0, -5, materials[0], cells, cell_size)
    procs = []
    for i in range(6):
        test = Test(14+i, -5, materials[i], cells, cell_size)
        proc = Process(target=runtest, args=(test, folder))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()


TCC = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 25, 50, 100, 250, 500]


def TestThermalCond():
    folder = InitFolder("TestThermalCond")
    material = materials[1]._replace(thermal_cond=TCC[0])
    test = Test(10, -5, material, cells, cell_size)
    procs = []
    for i in range(len(TCC)):
        material = materials[1]._replace(thermal_cond=TCC[i])
        test = Test(2 + i, -6, material, cells, cell_size)
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
    return folder
