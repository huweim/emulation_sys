pip uninstall fp8 -y
rm -rf build dist fp8.egg-info
rm scaled_fp8_ops.cpython-312-x86_64-linux-gnu.so
pip install -e .