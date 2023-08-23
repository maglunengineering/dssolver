from typing import Dict, Callable

class DSSPlugin:
    settings = {}
    instantiate = False
    def __init__(self, owner:'DSS'):
        self.dss: 'DSS' = owner

    @classmethod
    def load_plugin(cls, owner:'DSS'):
        owner.plugins[cls] = DSSPlugin(owner) # Placeholder for plugins that are not instantiated
        if cls.instantiate:
            instance = cls(owner)
            owner.plugins[cls] = instance
            instance.load_instance()

    @classmethod
    def get_settings(cls) -> Dict[str, bool]:
        return cls.settings

    @classmethod
    def set_setting(cls, key, value):
        cls.settings[key] = value

    def load_instance(self):
        pass

    def on_after_dss_built(self):
        pass

    def get_functions(self) -> Dict[str, Callable]:
        return {}
