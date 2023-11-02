import os
import shutil


def clear_dir(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)


if __name__ == "__main__":
    clear_dir('/home/pc/main/rust/ventilator-dist/logs')
    clear_dir('/home/pc/main/rust/ventilator-dist/soln/intermediate')
    clear_dir('/home/pc/main/rust/ventilator-dist/soln/excel_logs')
