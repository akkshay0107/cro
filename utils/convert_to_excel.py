from pathlib import Path
import numpy as np
import pandas as pd

C = [125.0, 181.0, 39.0]


def convert_soln(filepath):
    with open(filepath, 'r') as f:
        arrs = [eval(x) for x in f.read().split("+")]
    assert len(C) == len(arrs)
    q = [np.array(a) for a in arrs]
    m = [np.ceil(q[i] / C[i]).astype(int) for i in range(len(C))]
    e = [(np.sum(q[i], axis=2) > 0).astype(np.int8) for i in range(len(C))]
    # log matrices out to excel
    outfile = "/home/pc/main/rust/network-cro/soln/excel_logs/" + Path(filepath).stem + ".xlsx"
    with pd.ExcelWriter(outfile) as writer:
        for i in range(len(C)):
            df1 = pd.DataFrame(q[i][:, :, 0])
            df2 = pd.DataFrame(m[i][:, :, 0])
            df3 = pd.DataFrame(e[i])
            df1.to_excel(writer, sheet_name="Sheet" + str(i))
            df2.to_excel(writer, sheet_name="Sheet" + str(i + 3))
            df3.to_excel(writer, sheet_name="Sheet" + str(i + 6))


if __name__ == "__main__":
    pathlist = Path("/home/pc/main/rust/network-cro/soln/intermediate").glob('*')
    for path in pathlist:
        convert_soln(path)
