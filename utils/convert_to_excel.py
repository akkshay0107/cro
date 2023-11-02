from pathlib import Path
import numpy as np
import pandas as pd

C = [[125.0, 78.0, 39.0], [181.0, 116.0, 24.0], [125.0, 78.0, 39.0]]


def convert_soln(filepath):
    with open(filepath, 'r') as f:
        arrs = [eval(x) for x in f.read().split("+")]
    assert len(C) == len(arrs)
    q = [np.array(a) for a in arrs]
    m = []
    for i in range(len(q)):
        arr = q[i].copy().astype(np.float64)
        arr[:, :, 0] /= C[i][0]
        arr[:, :, 0] = np.ceil(arr[:, :, 0])
        arr[:, :, 1] /= C[i][1]
        arr[:, :, 1] = np.ceil(arr[:, :, 1])
        arr[:, :, 2] /= C[i][2]
        arr[:, :, 2] = np.ceil(arr[:, :, 2])
        m.append(arr.astype(np.int32))
    e = [(np.sum(q[i], axis=2) > 0).astype(np.int8) for i in range(len(C))]
    # log matrices out to excel
    outfile = "/home/pc/main/rust/ventilator-dist/soln/excel_logs/" + \
        Path(filepath).stem + ".xlsx"
    with pd.ExcelWriter(outfile) as writer:
        for i in range(len(C)):
            df1_1 = pd.DataFrame(q[i][:, :, 0])
            df1_2 = pd.DataFrame(q[i][:, :, 1])
            df1_3 = pd.DataFrame(q[i][:, :, 2])
            df2_1 = pd.DataFrame(m[i][:, :, 0])
            df2_2 = pd.DataFrame(m[i][:, :, 1])
            df2_3 = pd.DataFrame(m[i][:, :, 2])
            df3_1 = pd.DataFrame(e[i])
            df1_1.to_excel(writer, sheet_name="Sheet" +
                           str(i), startrow=0, startcol=0)
            df2_1.to_excel(writer, sheet_name="Sheet" +
                           str(i + 3), startrow=0, startcol=0)
            df3_1.to_excel(writer, sheet_name="Sheet" +
                           str(i + 6), startrow=0, startcol=0)
            df1_2.to_excel(writer, sheet_name="Sheet" + str(i),
                           startrow=df1_1.shape[0]+3, startcol=0)
            df2_2.to_excel(writer, sheet_name="Sheet" + str(i + 3),
                           startrow=df2_1.shape[0]+3, startcol=0)
            df1_3.to_excel(writer, sheet_name="Sheet" + str(i),
                           startrow=df1_1.shape[0]+df1_2.shape[0]+6, startcol=0)
            df2_3.to_excel(writer, sheet_name="Sheet" + str(i + 3),
                           startrow=df1_1.shape[0]+df1_2.shape[0]+6, startcol=0)


if __name__ == "__main__":
    pathlist = Path(
        "/home/pc/main/rust/ventilator-dist/soln/intermediate").glob('*')
    for path in pathlist:
        convert_soln(path)
