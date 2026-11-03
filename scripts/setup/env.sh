# Genesis を GPU で動かすシェルで source する環境変数。
#   source scripts/setup/env.sh
# ROS 2 の /opt/ros/jazzy/setup.bash も LD_LIBRARY_PATH に追記するだけなので、source の順序は問わない。

# WSL の libcuda.so (バージョン無し名) は ldconfig に載らず、Genesis (quadrants) が CUDA を検出できない
_wsl_cuda_lib_dir="/usr/lib/wsl/lib"
if [[ ! -e "${_wsl_cuda_lib_dir}/libcuda.so" ]]; then
  echo "ERROR: ${_wsl_cuda_lib_dir}/libcuda.so が無い。Windows 側の NVIDIA ドライバ（WSL の GPU 対応）を確認する。" >&2
  unset _wsl_cuda_lib_dir
  return 1
fi
case ":${LD_LIBRARY_PATH:-}:" in
  *":${_wsl_cuda_lib_dir}:"*) ;;  # 二重 source しても重複追加しない
  *) export LD_LIBRARY_PATH="${_wsl_cuda_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
unset _wsl_cuda_lib_dir
