import hashlib
from typing import List

from yacs.config import CfgNode


def get_file_md5(files_name: List[str]) -> str:
    m = hashlib.md5()
    for f_name in files_name:
        with open(f_name, 'rb') as f:
            while True:
                data = f.read(4096)
                if not data:
                    break
                m.update(data)

    return m.hexdigest()



if __name__ == '__main__':
    file_name = r" "
    file_md5 = get_file_md5([file_name])
    print(file_md5 * 3)
    md5_name = hashlib.md5((file_md5 * 3).encode()).hexdigest()

    print(md5_name)
