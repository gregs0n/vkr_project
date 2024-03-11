from collections import namedtuple

Material = namedtuple("Material", ["name", "thermal_cond", "tmax", "tmin"])

Test = namedtuple(
    "Test",
    ["test_no", "bnd", "material", "cells", "cell_size", "limits"],
    defaults=[[1, 1]],
)


def getTestName(test: Test) -> str:
    step = test.limits[0] / ((test.cells[0] * (test.cell_size - 1)))
    words = [
        f"test_{test.test_no:03d}.{test.bnd:02d}",
        "{0:_>21}".format(f"{test.material.name}[{test.material.thermal_cond:06.2f}]"),
        f"step={step:.6e}",
        f"{repr(test.cells)}",
        f"{test.cell_size:02d}",
    ]
    return "_".join(words)


materials = [
    Material(name="Acrylic_glass", thermal_cond=0.2, tmax=433, tmin=293),
    Material(name="My_material", thermal_cond=1.0, tmax=600, tmin=300),  # 1500 312
    Material(name="Manganese", thermal_cond=7.81, tmax=1519, tmin=437),
    Material(name="Germanium", thermal_cond=60.2, tmax=1211.4, tmin=632),
    Material(name="Aluminium", thermal_cond=237.0, tmax=933.47, tmin=627),
    Material(name="Copper", thermal_cond=401.0, tmax=1357.77, tmin=868),
]

if __name__ == "__main__":
    for i in range(6):
        t = Test(4, 8, materials[i], [10, 10], 5)
        print(getTestName(t))
