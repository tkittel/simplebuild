# Installation step of sbldpy modules (pymod_* subdirs), invoked from the
# makefiles after linking the module library. It generates the Python file of
# the module, and (unless disabled with the undocumented environment variable
# SIMPLEBUILD_SBLDPY_TYPESTUBS=0) loads the library to check the definition of
# the module and generate .pyi type stubs. The environment variable is checked
# here (rather than when generating the makefiles), so it takes effect
# immediately, e.g. after failures to load modules at build time.

def typestubs_enabled():
    import os
    v = os.environ.get('SIMPLEBUILD_SBLDPY_TYPESTUBS') or '1'
    return v.lower() not in ('false', 'off', 'no', '0')

def main():
    import sys
    import pathlib
    import importlib.util
    runtime, lib, modname, pyfile, pyifile = sys.argv[1:]
    #Use the runtime module installed for the build:
    spec = importlib.util.spec_from_file_location( '_sbldpy_runtime_for_install', runtime )
    rt = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( rt )
    if typestubs_enabled():
        ec = rt.main( [ 'install', lib, modname, pyfile, pyifile ] )
    else:
        ec = rt.main( [ 'install', lib, modname, pyfile ] )
        pathlib.Path( pyifile ).unlink( missing_ok = True )
    sys.exit( ec )

if __name__ == '__main__':
    main()
