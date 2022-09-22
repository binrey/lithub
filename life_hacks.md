- [Remote jupyter notebooks with ssh port forwarding](https://thedatafrog.com/en/articles/remote-jupyter-notebooks/)    
- [SSH-туннели — пробрасываем порт](https://habr.com/ru/post/81607/)    
- [ПОДКЛЮЧЕНИЕ И НАСТРОЙКА SSHFS В LINUX](https://losst.ru/podklyuchenie-i-nastrojka-sshfs-v-linux)
- [How to set up Anaconda and Jupyter Notebook the right way](https://towardsdatascience.com/how-to-set-up-anaconda-and-jupyter-notebook-the-right-way-de3b7623ea4a)
- Add kernel to jupyter notebook started from another kernel: 

```
$conda activate new-env
$conda deactivate
$conda install -c conda-forge nb_conda_kernels
optional:
$python -m ipykernel install --user --name new-env --display-name new-env-kernel
```
- Jupyter Notebook Extentions:
```
pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install --user
```
- Test TensorRT
```
/usr/src/tensorrt/bin/trtexec --loadEngine=path/to/engine --batch=1 --warmUp=5000 --duration=1 --iterations=1000 --streams=1
```
