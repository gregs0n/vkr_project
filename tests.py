from scheme import BalanceScheme
from enviroment import *
from time import strftime

def runtest(test: Test, show_plot: bool):
    print('-'*98)
    u0 = None
    aruntime = 0
    if 1:
        aStep = 1/20
        aCells = 5
        aCellSize = int(test.limits[0]/(aStep*aCells))+1
        assumpTest = Test(0, test.bnd, test.material,
                          aStep, cells=[aCells, aCells], cell_size=aCellSize)
        assump = BalanceScheme(assumpTest)
        print("Assumption for: " + getTestName(test))
        u0, aruntime = assump.Compute(1e-3)
        u0 *= 0.01
        #assump.show_res(code=0, show_plot=1)
    
    runtime = 0
    s = BalanceScheme(test)
    #s.show_res(3, show_plot=1)
    print(s.test_name)
    _, runtime = s.Compute(1e-3, u0)
    print(s.U.max(), s.U.min())
    print('\n')
    #s.trace_newt_err(show_plot=1)
    #s.trace_cg_err(show_plot=show_plot)
    s.show_res(show_plot=show_plot)
    return runtime, aruntime

step = 1/100
cell = 5

def SingleTest(debug: bool, show_plot: bool):
    t1, t2 = 0, 0
    filename = 'logs/SingleTest_log.txt'
    test = Test(666, -4, materials[1], step, [cell, cell], int(1/(step*cell))+1)

    file = open(filename, 'a')
    print(f"\nTest '{getTestName(test)}'\nStarted - {strftime('%H:%M:%S')}", file=file)
    t1, t2 = runtest(test, show_plot)
    print(f"{t1:.6f}, {t2:.6f}", file=file)
    print(f"Test's over - {strftime('%H:%M:%S')}", file=file)
    file.close()

def TestMaterials(debug: bool, show_plot: bool):
    t1, t2 = 0, 0
    filename = 'logs/TestMaterials_log.txt'
    if debug:
        file = open(filename, 'a')
        print(f"\nTest started - {strftime('%H:%M:%S')}", file=file)
        file.close()
    for i in range(6):
        test = Test(i, -4, materials[i], step, [cell, cell], int(1/(step*cell))+1)
        t1, t2 = runtest(test, show_plot)
        if debug:
            file = open(filename, 'a')
            print(f"{t1:.6f}, {t2:.6f}", file=file)
            file.close()
    if debug:
        file = open(filename, 'a')
        print(f"Test's over - {strftime('%H:%M:%S')}", file=file)
        file.close()

TCC = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 25, 50, 100, 250, 500]
def TestThermalCond(debug: bool, show_plot: bool):
    t1, t2 = 0, 0
    filename = 'logs/TestThermalCond_log.txt'
    if debug:
        file = open(filename, 'a')
        print(f"\nTest started - {strftime('%H:%M:%S')}", file=file)
        file.close()
    for i in range(len(TCC)):
        material = materials[1]._replace(thermal_cond=TCC[i])
        test = Test(10+i, -2, material, step, [cell, cell], int(1/(step*cell))+1)
        t1, t2 = runtest(test, show_plot)
        if debug:
            file = open(filename, 'a')
            print(f"{t1:.6f}, {t2:.6f}", file=file)
            file.close()
    if debug:
        file = open(filename, 'a')
        print(f"Test's over - {strftime('%H:%M:%S')}", file=file)
        file.close()

def TestMyFunctions(debug: bool, show_plot: bool):
    t1, t2 = 0, 0
    filename = 'logs/TestMyFunctions_log.txt'
    if debug:
        file = open(filename, 'a')
        print(f"\nTest started - {strftime('%H:%M:%S')}", file=file)
        file.close()
    for i in range(6):
        test = Test(40+i, i, materials[1], step, [cell, cell], int(1/(step*cell))+1)
        t1, t2 = runtest(test, show_plot)
        if debug:
            file = open(filename, 'a')
            print(f"{t1:.6f}, {t2:.6f}", file=file)
            file.close()
    if debug:
        file = open(filename, 'a')
        print(f"Test's over - {strftime('%H:%M:%S')}", file=file)
        file.close()