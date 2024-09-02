from datetime import datetime
from os import mkdir
from multiprocessing import Process, Pool

from scheme import BalanceScheme
from enviroment import *
from boundary import getBoundary
from custom_numerics.wraps import timer

@timer
def runtest(data: tuple): #test: Test, folder=""
    test, folder = data
    print(f"{datetime.now().strftime('%H:%M:%S')} - |[ {test.test_no:03} ]| started {getTestName(test)}")
    u0 = None
    eps = 1e-6

    params = [
        *getBoundary(test, f_off=0, g_off=0),
        test.material,
        test.limits,
        test.cells,
        test.cell_size,
        str(test.test_no),#getTestName(test),
        folder
    ]

    s = BalanceScheme(*params)
    s.show_res(code=1, show_plot=1)
    s.show_res(code=2, show_plot=1)
    _, exit_code = s.Compute(eps, u0)
    print(s.U.max())
    #s.show_res(code=1, show_plot=0)
    s.show_res(code=3, show_plot=1)
    #s.save()
    print(f"{datetime.now().strftime('%H:%M:%S')} - |[ {test.test_no:03} ]|    over {s.test_name}")
    return exit_code

cells = [10, 10]
cell_size = 11


def SingleTest():
    folder = InitFolder("SingleTest")
    material = materials[1]._replace(thermal_cond=10.0)
    test = Test(45, -1, material, cells, cell_size)

    runtest((test, folder))


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
    
def TestSquares(input_start, input_end, tcc_loc, start_cell_size=9):
    folder = InitFolder("TestSquares")
    material = materials[1]._replace(thermal_cond=tcc_loc)
    tasks = []
    procs = []
    #pool = Pool(4)
    for cell in range(input_start, input_end):
        cell_size = start_cell_size
        exit_code = 1
        while cell_size > 2 and exit_code == 1:
            if cell_size < start_cell_size:
                print(f"RESTART WITH REDUCED CELL SIZE - {cell_size}")
            test = Test(cell, -1, material, [cell, cell], cell_size)
            tasks.append((test, folder))
            exit_code = runtest((test, folder))
            #if exit_code == 1 and input("restart? y/n:") != 'y':
            #    break
            cell_size -= 1
        continue
        #proc = Process(target=runtest, args=(test, folder))
        #procs.append(proc)
        #proc.start()
    #for proc in procs:
    #    proc.join()
    #pool.map(runtest, tasks)


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
    mkdir(folder + "/bin")
    return folder
