from scheme import BalanceScheme
from enviroment import *
from time import strftime
from boundary import getBoundary

def runtest(test: Test):
    print('-'*98)
    u0 = None
    aruntime = 0
    if 0:
        aCells = 10
        aCellSize = 4
        assumpTest = Test(0, test.bnd, test.material,
                          cells=[aCells, aCells], cell_size=aCellSize)
        params = [*getBoundary(assumpTest, f_off=0, g_off=0),
                  assumpTest.material,
                  assumpTest.limits,
                  assumpTest.cells, aCellSize,
                  getTestName(assumpTest)]
        assump = BalanceScheme(*params)
        print("Assumption for: " + getTestName(test))
        u0, aruntime = assump.Compute(1e-3)
        u0 *= 0.01
        #assump.show_res(code=0, show_plot=1)
    
    runtime = 0
    params = [*getBoundary(test, f_off=0, g_off=0),
              test.material,
              test.limits, test.cells, test.cell_size,
              getTestName(test)]
    s = BalanceScheme(*params)
    #s.show_res(0, show_plot=1)
    #s.show_res(2, show_plot=1)
    print(s.test_name)
    _, runtime = s.Compute(1e-3, u0)
    print(s.U.max(), s.U.min())
    print('\n')
    #s.trace_newt_err(show_plot=1)
    #s.trace_cg_err(show_plot=1)
    s.show_res(show_plot=1)
    return runtime, aruntime

cells = [1, 1]
cell_size = 11

def SingleTest():
    t1, t2 = 0, 0
    filename = 'logs/SingleTest_log.txt'
    test = Test(666, -4, materials[1], cells, cell_size)

    LogTest(filename, 0, test)
    t1, t2 = runtest(test)
    LogTest(filename, 1, t1, t2)
    LogTest(filename, 2)

def TestMaterials():
    t1, t2 = 0, 0
    filename = 'logs/TestMaterials_log.txt'
    test = Test(0, -4, materials[0], cells, cell_size)
    LogTest(filename, 0, test)
    for i in range(6):
        test = Test(i, -4, materials[i], cells, cell_size)
        t1, t2 = runtest(test)
        LogTest(filename, 1, t1, t2)
    LogTest(filename, 2)

TCC = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10, 25, 50, 100, 250, 500]
def TestThermalCond():
    t1, t2 = 0, 0
    filename = 'logs/TestThermalCond_log.txt'
    material = materials[1]._replace(thermal_cond=TCC[0])
    test = Test(10, -3, material, cells, cell_size)
    LogTest(filename, 0, test)
    for i in range(len(TCC)):
        material = materials[1]._replace(thermal_cond=TCC[i])
        test = Test(10+i, -3, material, cells, cell_size)
        t1, t2 = runtest(test)
        LogTest(filename, 1, t1, t2)
    LogTest(filename, 2)

def TestMyFunctions():
    t1, t2 = 0, 0
    filename = 'logs/TestMyFunctions_log.txt'
    test = Test(40, 0, materials[1], cells, cell_size)
    LogTest(filename, 0, test)
    for i in range(6):
        test = Test(40+i, i, materials[1], cells, cell_size)
        t1, t2 = runtest(test)
        LogTest(filename, 1, t1, t2)
    LogTest(filename, 2)

def LogTest(fname: str, cmd: int, *args):
    file = open(fname, 'a')
    if cmd == 0:
        print(f"\nTest '{getTestName(args[0])}'", file=file)
        print(f"Started - {strftime('%H:%M:%S')}", file=file)
    elif cmd == 1:
        print(f"{args[0]:.6f}, {args[1]:.6f}", file=file)
    elif cmd == 2:
        print(f"Test's over - {strftime('%H:%M:%S')}", file=file)
    file.close()
