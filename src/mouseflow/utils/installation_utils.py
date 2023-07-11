from importlib import import_module


def is_installed(module_name: str) -> bool:
    try:
        import_module(module_name)
        return True
    except ImportError as e:
        return False