from typing import Tuple, Iterable


_settings = {}

def add_setting(k:str, v:object) -> None:
    _settings[k] = v

def get_setting(k:str, default:object) -> object:
    return _settings.setdefault(k, default)

def set_setting(key:str, val:object):
    _settings[key] = val

def get_by_category(category) -> Iterable[Tuple[str, object]]:
    return ((k,v) for k,v in _settings.items() if k.startswith(category))
