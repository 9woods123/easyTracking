pip install Cython
git clone -b 0.5.6 https://github.com/msgpack/msgpack-python.git
cd msgpack-python
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
python setup.py build_ext --inplace
python setup.py install
