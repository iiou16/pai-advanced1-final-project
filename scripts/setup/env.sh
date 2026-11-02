# Genesis を GPU で動かすシェルで source する環境変数。
#   source scripts/setup/env.sh
# ROS 2 を併用する場合は /opt/ros/jazzy/setup.bash を先に source してから読み込む。

# WSL の libcuda.so (バージョン無し名) は ldconfig に載らず、Genesis (quadrants) が CUDA を検出できない
_wsl_cuda_lib_dir="/usr/lib/wsl/lib"
case ":${LD_LIBRARY_PATH:-}:" in
  *":${_wsl_cuda_lib_dir}:"*) ;;  # 二重 source しても重複追加しない
  *) export LD_LIBRARY_PATH="${_wsl_cuda_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
unset _wsl_cuda_lib_dir
