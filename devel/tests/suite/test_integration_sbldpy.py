# Tests of sbldpy modules (pymod_* subdirs), which provide Python bindings via
# ctypes, without compiling against Python headers or linking with libpython.

import platform
import shutil
import sys

import pytest

from sbtestutils import require_feature, run, write_files

pytestmark = pytest.mark.integration

_datadir = __import__('pathlib').Path(__file__).parent / 'data_sbldpy'

_files = {
    'simplebuild.cfg' : '[bundle]\nname = "proj"\n',
    'PkgA/pkg.info' : 'package()\n',
    'PkgA/libinc/A.hh' : 'int a_value();\n',
    'PkgA/libsrc/A.cc' : '#include "PkgA/A.hh"\nint a_value() { return 17; }\n',
    'PkgA/python/purepy.py' : 'value = 42\n',
    'PkgB/pkg.info' : 'package(USEPKG PkgA)\n',
    'PkgB/pymod_small/mod.cc' : ( '#include "sbldpy/sbldpy.hh"\n'
                                  'SBLDPY_MODULE( small ) { m.def( "value", []() { return 1; } ); }\n' ),
}

def _have_numpy():
    try:
        import numpy # noqa F401
        return True
    except ImportError:
        return False

def inst( d ):
    return d / 'simplebuild_cache' / 'install'

@pytest.fixture( scope = 'module' )
def proj( tmp_path_factory ):
    d = tmp_path_factory.mktemp( 'sbldpy' ) / 'proj'
    write_files( d, _files )
    write_files( d, { 'PkgA/pymod_feat/feat.cc' : ( _datadir / 'feat.cc' ).read_text() } )
    run( 'sb', d )
    return d

def check( d, *sections ):
    res = run( [ 'sbenv', sys.executable, _datadir / 'check.py' ] + list( sections ), d )
    for s in sections:
        assert f'check {s} OK' in res.stdout
    assert not res.stderr.strip()

def test_installed_files( proj ):
    i = inst( proj )
    assert ( i / 'python' / 'sbldpy.py' ).is_file()
    assert ( i / 'sbldpy' / 'include' / 'sbldpy' / 'sbldpy.hh' ).is_file()
    for f in ( '_sbldpy_feat.so', 'feat.py', 'feat.pyi', '__init__.py', 'purepy.py' ):
        assert ( i / 'python' / 'PkgA' / f ).exists(), f
    for f in ( '_sbldpy_small.so', 'small.py', 'small.pyi', '__init__.py' ):
        assert ( i / 'python' / 'PkgB' / f ).exists(), f
    stub = ( i / 'python' / 'PkgA' / 'feat.py' ).read_text()
    assert "_sbldpy.load_module( globals(), '_sbldpy_feat' )" in stub
    pyi = ( i / 'python' / 'PkgA' / 'feat.pyi' ).read_text()
    for expected in ( 'def add(a: float, b: float = 1.5) -> float:',
                      'def iadd(arg0: int, arg1: int, /) -> int:',
                      'def integrate(f: collections.abc.Callable[[float], float], n: int = 1000) -> float:',
                      'def linspace(arg0: float, arg1: float, arg2: int, /) -> numpy.typing.NDArray[numpy.float32]:',
                      'class Counter:', '    @typing.overload', '    def increment(self, n: int = 1) -> int:',
                      '    @property', '    @value.setter', '    @staticmethod',
                      'pi: float', 'pair: tuple[int, float]' ):
        assert expected in pyi, expected
    #The .pyi file is valid Python:
    compile( pyi, 'feat.pyi', 'exec' )

def test_basic( proj ):
    check( proj, 'basic' )

def test_arrays( proj ):
    check( proj, 'arrays' )

def test_numpy( proj ):
    require_feature( 'numpy', _have_numpy() )
    check( proj, 'numpy' )

def test_classes( proj ):
    check( proj, 'classes' )

def test_callbacks( proj ):
    check( proj, 'callbacks' )

def test_other_module( proj ):
    res = run( [ 'sbenv', sys.executable, '-c',
                 'import PkgB.small, PkgA.purepy; print(PkgB.small.value(), PkgA.purepy.value)' ], proj )
    assert res.stdout.strip() == '1 42'

def test_no_python_dependency( proj ):
    #Not compiled with Python flags:
    mf = ( proj / 'simplebuild_cache' / 'bld' / 'makefiles' / 'Makefile_PkgA.make' ).read_text()
    lines = [ line for line in mf.splitlines() if 'pymod_feat/feat.cc' in line and ' -c ' in line ]
    assert len( lines ) == 1
    assert 'PYBIND11' not in lines[0] and 'ython' not in lines[0]
    assert '-fvisibility=hidden' in lines[0]
    lib = inst( proj ) / 'python' / 'PkgA' / '_sbldpy_feat.so'
    #Not linked with libpython:
    if platform.system() == 'Darwin':
        deps = run( [ 'otool', '-L', lib ], proj ).stdout
    else:
        deps = run( [ 'ldd', lib ], proj ).stdout
    assert 'libPKG__PkgA' in deps
    assert 'python' not in deps.lower()
    #Only the four sbldpy_* functions are exported (apart from symbols of the
    #standard library which are always exported):
    nm = shutil.which( 'nm' )
    if nm:
        args = [ nm, '-gU', lib ] if platform.system() == 'Darwin' else [ nm, '-D', '--defined-only', lib ]
        syms = set( line.split()[-1].lstrip( '_' ) for line in run( args, proj ).stdout.splitlines()
                    if line.strip() )
        assert set( s for s in syms if 'sbldpy' in s.lower() ) == set( [ 'sbldpy_abi_version',
                                                                         'sbldpy_module_info',
                                                                         'sbldpy_call',
                                                                         'sbldpy_last_error' ] )

def test_abi_mismatch( proj ):
    code = ( 'import sbldpy\nsbldpy.abi_version = 999\n'
             'try:\n    import PkgA.feat\nexcept ImportError as e:\n    print(e)\n' )
    res = run( [ 'sbenv', sys.executable, '-c', code ], proj )
    assert 'was built for sbldpy ABI version 1, but the sbldpy runtime' in res.stdout
    assert 'supports ABI version 999' in res.stdout and 'must be rebuilt' in res.stdout

def _build_errors( d, files, expected_msg ):
    write_files( d, files )
    res = run( 'sb', d, expect_ec = None )
    assert res.ec != 0
    assert expected_msg in res.out, res.out
    for f in files:
        p = d / f
        p.unlink()
        if not any( p.parent.iterdir() ):
            p.parent.rmdir()
    run( 'sb', d )

def test_build_errors( tmp_path ):
    d = tmp_path / 'proj'
    write_files( d, _files )
    run( 'sb', d )
    _mod = '#include "sbldpy/sbldpy.hh"\nSBLDPY_MODULE( %s ) {\n%s\n}\n'
    #Reserved and invalid names:
    _build_errors( d, { 'PkgA/python/_sbldx.py' : '' }, 'names starting with "_sbld" are reserved' )
    _build_errors( d, { 'PkgA/pycpp__sbldx/mod.cc' : '' }, 'names starting with "_sbld" are reserved' )
    _build_errors( d, { 'PkgA/pymod__x/mod.cc' : _mod%('_x','') }, 'Invalid name of sbldpy module directory' )
    _build_errors( d, { 'PkgA/pymod_class/mod.cc' : _mod%('class','') }, 'can not be a Python keyword' )
    _build_errors( d, { 'PkgA/pymod_purepy/mod.cc' : _mod%('purepy','') },
                   'Python module PkgA.purepy is provided in both sbldpy (pymod_purepy/) and pure'
                   ' (python/purepy.py) forms' )
    _build_errors( d, { 'PkgB/pymod_small2/mod.cc' : _mod%('small2',''),
                        'PkgB/pycpp_small2/mod.cc' : '' },
                   'Python module PkgB.small2 is provided in both' )
    _build_errors( d, { 'Sbldpy/pkg.info' : 'package()\n' }, 'Invalid package name: "Sbldpy"' )
    #Errors in the C++ code detected at compile time:
    _build_errors( d, { 'PkgA/pymod_x/mod.cc' : _mod%('y','') },
                   'name in SBLDPY_MODULE(..) must match the name of the module' )
    _build_errors( d, { 'PkgA/pymod_x/mod.cc' : _mod%('x','m.def("f",[](float){});') },
                   'float is not supported as parameter type (use double)' )
    _build_errors( d, { 'PkgA/pymod_x/mod.cc' : _mod%('x','m.def("f",[](int){});') },
                   'integers can only be received as std::int64_t' )
    _build_errors( d, { 'PkgA/pymod_x/mod.cc' : _mod%('x','m.def("f",[](){ return std::vector<double>(); });') },
                   'std::vector can not be returned directly' )
    _build_errors( d, { 'PkgA/pymod_x/mod.cc' : _mod%('x','m.def("f",[](){ static double x; return &x; });') },
                   'pointers can not be returned' )
    #Errors detected when the module is loaded at build time:
    _build_errors( d, { 'PkgA/pymod_x/mod.cc' : _mod%('x','m.def("f",[](std::int64_t){},{sbldpy::arg("a","notanint")});') },
                   'Invalid default value of argument "a" of f: expected int, got str' )
    _build_errors( d, { 'PkgA/pymod_x/mod.cc' : _mod%('x','m.def("f",[](std::int64_t){},{"a","b"});') },
                   'wrong number of argument names given for f' )
    _build_errors( d, { 'PkgA/pymod_x/mod.cc' : _mod%('x','m.def("f",[](){});m.def("f",[](){});') },
                   'Duplicate name in module PkgA.x: f' )
    _build_errors( d, { 'PkgA/pymod_x/mod.cc' : ( '#include "sbldpy/sbldpy.hh"\nstruct S{ S(std::int64_t){} };\n'
                                                  'SBLDPY_MODULE( x ) { m.add_class<S>("S").def_init<std::int64_t>()'
                                                  '.def_init([](std::int64_t a){ return S(a); }); }\n' ) },
                   'Constructors of class S can not be distinguished by their number of arguments' )
    _build_errors( d, { 'PkgA/pymod_x/mod.cc' : ( '#include "sbldpy/sbldpy.hh"\nstruct S{};\n'
                                                  'SBLDPY_MODULE( x ) { m.def("f",[](const S&){}); }\n' ) },
                   'a class which was not (yet) added to the module with add_class' )

def test_deinstall_and_typestubs_switch( tmp_path ):
    d = tmp_path / 'proj'
    write_files( d, _files )
    run( 'sb', d )
    pydir = inst( d ) / 'python' / 'PkgB'
    assert ( pydir / 'small.pyi' ).is_file()
    #The type stubs (and loading of the module at build time) can be disabled
    #with an undocumented environment variable, which takes effect when the
    #module is rebuilt:
    src = d / 'PkgB' / 'pymod_small' / 'mod.cc'
    src.write_text( src.read_text() + '//changed\n' )
    run( 'sb', d, env_extra = { 'SIMPLEBUILD_SBLDPY_TYPESTUBS' : '0' } )
    assert ( pydir / 'small.py' ).is_file() and not ( pydir / 'small.pyi' ).exists()
    #Can be used to work around problems when loading modules at build time:
    src.write_text( src.read_text().replace( 'm.def(', 'm.def( "dup", [](){} ); m.def( "dup", [](){} ); m.def(' ) )
    res = run( 'sb', d, expect_ec = None )
    assert res.ec != 0 and 'Duplicate name in module PkgB.small: dup' in res.out
    run( 'sb', d, env_extra = { 'SIMPLEBUILD_SBLDPY_TYPESTUBS' : 'off' } )
    assert not ( pydir / 'small.pyi' ).exists()
    src.write_text( src.read_text().replace( 'm.def( "dup", [](){} ); m.def( "dup", [](){} ); ', '' ) )
    run( 'sb', d )
    assert ( pydir / 'small.pyi' ).is_file()
    assert run( 'sbenv python3 -c "import PkgB.small;print(PkgB.small.value())"', d ).stdout.strip() == '1'
    #Removing a module removes all its installed files:
    shutil.rmtree( d / 'PkgB' / 'pymod_small' )
    res = run( 'sb', d )
    assert not pydir.exists()
    ( d / 'PkgB' / 'python' ).mkdir()
    ( d / 'PkgB' / 'python' / 'other.py' ).write_text( '' )
    res = run( 'sb', d )
    assert sorted( f.name for f in pydir.iterdir() ) == [ '__init__.py', 'other.py' ]
    del res
