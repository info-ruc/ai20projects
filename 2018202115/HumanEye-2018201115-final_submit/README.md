name:		邓适
student ID:	2018202115

###file folders intro

slides：		展示用的ppt以及思维导图

codes:			整个项目的代码，不包含模型训练结果、中间输出、测试结果以及数据集

test_result:	训练结果和测试结果

BLEU_result:	生成的结果评分文件

debug_notes:	debug记录

references:		参考文献

###test our project

为方便运行测试，我们向inlab实验室借了一个服务器账号，可以用于运行该项目。

IP:			10.10.16.32(仅供校内访问)
user ID:	txg
password:	txg_inlab123

Enter the relevant folder:
```shell
	cd image_captioning-master/
```

To activate envs:
```shell
	conda activate image
```

To test:
```shell
	python main.py --phase=test     --model_file='./models/282450.npy'     --beam_size=3
```
282450.npy is the model we have trained
The generated captions will be saved in the folder `test/results`.

###envs
# packages in environment at /home/txg/miniconda3/envs/image:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main
_tflow_select             2.1.0                       gpu
absl-py                   0.11.0           py37h06a4308_0
astor                     0.8.1                    py37_0
blas                      1.0                         mkl
bzip2                     1.0.8                h7b6447c_0
c-ares                    1.16.1               h7b6447c_0
ca-certificates           2020.10.14                    0
cairo                     1.14.12              h8948797_3
certifi                   2020.6.20          pyhd3eb1b0_3
click                     7.1.2                      py_0
cloudpickle               1.6.0                      py_0
cudatoolkit               10.1.243             h6bb024c_0
cudnn                     7.6.5                cuda10.1_0
cupti                     10.1.168                      0
cycler                    0.10.0                   py37_0
cytoolz                   0.11.0           py37h7b6447c_0
dask-core                 2.30.0                     py_0
dbus                      1.13.18              hb2f20db_0
decorator                 4.4.2                      py_0
expat                     2.2.10               he6710b0_2
ffmpeg                    4.0                  hcdf2ecd_0
fontconfig                2.13.0               h9420a91_0
freeglut                  3.0.0                hf484d3e_5
freetype                  2.10.4               h5ab3b9f_0
gast                      0.2.2                    py37_0
glib                      2.66.1               h92f7085_0
google-pasta              0.2.0                      py_0
graphite2                 1.3.14               h23475e2_0
grpcio                    1.31.0           py37hf8bcb03_0
gst-plugins-base          1.14.0               hbbd80ab_1
gstreamer                 1.14.0               hb31296c_0
h5py                      2.8.0            py37h989c5e5_3
harfbuzz                  1.8.8                hffaf4a1_0
hdf5                      1.10.2               hba1933b_1
icu                       58.2                 he6710b0_3
imageio                   2.9.0                      py_0
importlib-metadata        2.0.0                      py_1
intel-openmp              2020.2                      254
jasper                    2.0.14               h07fcdf6_1
joblib                    0.17.0                     py_0
jpeg                      9b                   h024ee3a_2
keras-applications        1.0.8                      py_1
keras-preprocessing       1.1.0                      py_1
kiwisolver                1.3.0            py37h2531618_0
lcms2                     2.11                 h396b838_0
ld_impl_linux-64          2.33.1               h53a641e_7
libedit                   3.1.20191231         h14c3975_1
libffi                    3.3                  he6710b0_2
libgcc-ng                 9.1.0                hdf63c60_0
libgfortran-ng            7.3.0                hdf63c60_0
libglu                    9.0.0                hf484d3e_1
libopencv                 3.4.2                hb342d67_1
libopus                   1.3.1                h7b6447c_0
libpng                    1.6.37               hbc83047_0
libprotobuf               3.13.0.1             hd408876_0
libstdcxx-ng              9.1.0                hdf63c60_0
libtiff                   4.1.0                h2733197_1
libuuid                   1.0.3                h1bed415_2
libvpx                    1.7.0                h439df22_0
libxcb                    1.14                 h7b6447c_0
libxml2                   2.9.10               hb55368b_3
lz4-c                     1.9.2                heb0550a_3
markdown                  3.3.3            py37h06a4308_0
matplotlib                3.3.2                         0
matplotlib-base           3.3.2            py37h817c723_0
mkl                       2020.2                      256
mkl-service               2.3.0            py37he904b0f_0
mkl_fft                   1.2.0            py37h23d657b_0
mkl_random                1.1.1            py37h0573a6f_0
ncurses                   6.2                  he6710b0_1
networkx                  2.5                        py_0
nltk                      3.5                        py_0
numpy                     1.16.2           py37h7e9f1db_0
numpy-base                1.16.2           py37hde5b4d6_0
olefile                   0.46                     py37_0
opencv                    3.4.2            py37h6fd60c2_1
openssl                   1.1.1h               h7b6447c_0
pandas                    1.1.3            py37he6710b0_0
pcre                      8.44                 he6710b0_0
pillow                    8.0.1            py37he98fc37_0
pip                       20.2.4           py37h06a4308_0
pixman                    0.40.0               h7b6447c_0
protobuf                  3.13.0.1         py37he6710b0_1
py-opencv                 3.4.2            py37hb342d67_1
pyparsing                 2.4.7                      py_0
pyqt                      5.9.2            py37h05f1152_2
python                    3.7.9                h7579374_0
python-dateutil           2.8.1                      py_0
pytz                      2020.1                     py_0
pywavelets                1.1.1            py37h7b6447c_2
pyyaml                    5.3.1            py37h7b6447c_1
qt                        5.9.7                h5867ecd_1
readline                  8.0                  h7b6447c_0
regex                     2020.10.15       py37h7b6447c_0
scikit-image              0.16.2           py37h0573a6f_0
scipy                     1.5.2            py37h0b6359f_0
setuptools                50.3.1           py37h06a4308_1
sip                       4.19.8           py37hf484d3e_0
six                       1.15.0           py37h06a4308_0
sqlite                    3.33.0               h62c20be_0
tensorboard               1.14.0           py37hf484d3e_0
tensorflow                1.14.0          gpu_py37h74c33d7_0
tensorflow-base           1.14.0          gpu_py37he45bfe2_0
tensorflow-estimator      1.14.0                     py_0
tensorflow-gpu            1.14.0               h0d30ee6_0
termcolor                 1.1.0                    py37_1
tk                        8.6.10               hbc83047_0
toolz                     0.11.1                     py_0
tornado                   6.0.4            py37h7b6447c_1
tqdm                      4.51.0             pyhd3eb1b0_0
werkzeug                  1.0.1                      py_0
wheel                     0.35.1             pyhd3eb1b0_0
wrapt                     1.12.1           py37h7b6447c_1
xz                        5.2.5                h7b6447c_0
yaml                      0.2.5                h7b6447c_0
zipp                      3.4.0              pyhd3eb1b0_0
zlib                      1.2.11               h7b6447c_3
zstd                      1.4.5                h9ceee32_0















