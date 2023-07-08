def is_installed(module_name: str) -> bool:
    from importlib import import_module
    try:
        import_module(module_name)
        return True
    except ImportError as e:
        return False