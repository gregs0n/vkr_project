from scheme import BalanceScheme
from enviroment import *
from time import strftime

def runtest(test: Test):
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
    s.show_res(show_plot=1)
    return runtime, aruntime

step = 1/100
cell = 5

def SingleTest():
    t1, t2 = 0, 0
    filename = 'logs/SingleTest_log.txt'
    test = Test(666, -4, materials[1], step, [cell, cell], int(1/(step*cell))+1)

    LogTest(filename, 0, test)
    t1, t2 = runtest(test)
    LogTest(filename, 1, t1, t2)
    LogTest(filename, 2)

def TestMaterials():
    t1, t2 = 0, 0
    filename = 'logs/TestMaterials_log.txt'
    LogTest(filename, 0, test)
    for i in range(6):
        test = Test(i, -4, materials[i], step, [cell, cell], int(1/(step*cell))+1)
        t1, t2 = runtest(test)
        LogTest(filename, 1, t1, t2)
    LogTest(filename, 2)

TCC = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 25, 50, 100, 250, 500]
def TestThermalCond():
    t1, t2 = 0, 0
    filename = 'logs/TestThermalCond_log.txt'
    LogTest(filename, 0, test)
    for i in range(len(TCC)):
        material = materials[1]._replace(thermal_cond=TCC[i])
        test = Test(10+i, -2, material, step, [cell, cell], int(1/(step*cell))+1)
        t1, t2 = runtest(test)
        LogTest(filename, 1, t1, t2)
    LogTest(filename, 2)

def TestMyFunctions():
    t1, t2 = 0, 0
    filename = 'logs/TestMyFunctions_log.txt'
    LogTest(filename, 0, test)
    for i in range(6):
        test = Test(40+i, i, materials[1], step, [cell, cell], int(1/(step*cell))+1)
        t1, t2 = runtest(test)
        LogTest(filename, 1, t1, t2)
    LogTest(filename, 2)

def LogTest(fname: str, cmd: int, *args):
    file = open(fname, 'a')
    if cmd == 0:
        print(f"Test '{getTestName(args[0])}'", file=file)
        print(f"Started - {strftime('%H:%M:%S')}", file=file)
    elif cmd == 1:
        print(f"{args[0]:.6f}, {args[1]:.6f}", file=file)
    elif cmd == 2:
        print(f"Test's over - {strftime('%H:%M:%S')}", file=file)
    file.close()