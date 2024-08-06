from tqdm import tqdm
from typing import Dict, Any, NoReturn, Optional


class Pbar:
    def __init__(self, verbose: bool = False, total: int = None, desc: str = None, initial: int = 0):
        self.verbose = verbose
        self.pbar = tqdm(initial=initial, desc=desc, total=total, dynamic_ncols=False) if verbose else None
        self.__update = self.__update_pbar if verbose else self.__update_none
        self.info = {}

    def __update_pbar(self, info: Dict[str, Any], *args, **kwargs):
        state = kwargs.pop("state", "")
        if state == "add":
            for k, v in info:
                if k in self.info:
                    self.info[k] += v
                else:
                    self.info[k] = v
        elif state == "cover":
            self.info = info
        else:
            self.info = {}
        self.pbar.set_postfix(self.info)
        self.pbar.update()

    def __update_none(self, *args, **kwargs):
        ...

    def __call__(self, info: Optional[Dict[str, Any]] = None, *args, **kwargs) -> NoReturn:
        self.__update(info or {}, *args, **kwargs)

    def __del__(self):
        del self.pbar
